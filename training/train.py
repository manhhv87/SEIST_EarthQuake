"""
Training module for seismic neural network models.

This module implements the training loop and related utilities for seismic models,
including data loading, metric computation, learning rate scheduling, checkpointing,
and distributed training setup. It supports AMP (Automatic Mixed Precision),
TensorBoard logging, and early stopping.

Functions:
    train: Executes one training epoch over the training dataset.
    train_worker: Orchestrates the full training pipeline including validation and saving.
"""

import datetime
import inspect
import math
import os
import shutil
from typing import Union
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from config import Config
from models import create_model, load_checkpoint, save_checkpoint
from utils import *
from .postprocess import process_outputs
from .preprocess import SeismicDataset
from .validate import validate
import time


def train(
    args,
    tasks,
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    epoch,
    device,
    tensor_writer,
) -> Union[list, dict]:
    """
    Runs one training epoch for the provided model.

    Args:
        args: Configuration object containing training parameters.
        tasks (list): List of task names to train (e.g., 'p_arrival', 's_arrival').
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler.
        loss_fn (callable): Loss function.
        train_loader (DataLoader): DataLoader for training data.
        epoch (int): Current epoch number.
        device (torch.device): Device to perform training on.
        tensor_writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        list: Per-step training loss values.
        dict: Dictionary of Metrics objects with computed metrics per task.
    """
    model.train()

    train_loss_per_step = []
    average_meters = {}
    metrics_merged = {}
    sampling_rate = train_loader.dataset.sampling_rate()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for task in tasks:
        metrics = Metrics(
            task=task,
            metric_names=Config.get_metrics(task),
            sampling_rate=sampling_rate,
            time_threshold=args.time_threshold,
            num_samples=args.in_samples,
            device=device,
        )
        metrics_merged[f"{task}"] = metrics
        for metric in metrics.metric_names():
            average_meters[f"{task}_{metric}"] = AverageMeter(
                f"[{task.upper()}]{metric}", ":6.4f"
            )

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )

    (
        label_names,
        tgts_trans_for_loss,
        outs_trans_for_loss,
        outs_trans_for_res,
    ) = Config.get_model_config_(
        args.model_name,
        "labels",
        "targets_transform_for_loss",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )

    for step, (x, loss_targets, metrics_targets, _) in enumerate(train_loader):
        if isinstance(x, (list, tuple)):
            x = [xi.to(device) for xi in x]
        else:
            x = x.to(device)

        if isinstance(loss_targets, (list, tuple)):
            loss_targets = [yi.to(device) for yi in loss_targets]
        else:
            loss_targets = loss_targets.to(device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = model(x)
            outputs_for_loss = (
                outs_trans_for_loss(outputs)
                if outs_trans_for_loss is not None
                else outputs
            )
            loss_targets = (
                tgts_trans_for_loss(loss_targets)
                if tgts_trans_for_loss is not None
                else loss_targets
            )
            loss = loss_fn(outputs_for_loss, loss_targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]["lr"]

        step_batch_size = x[0].size(0) if isinstance(x, list) else x.size(0)

        if is_dist_avail_and_initialized():
            loss = reduce_tensor(loss, "AVG")
            step_batch_size = torch.tensor(
                step_batch_size, device=device, dtype=torch.int32
            )
            step_batch_size = reduce_tensor(step_batch_size)
            dist.barrier()
            step_batch_size = step_batch_size.item()

        average_meters["loss"].update(loss.item(), step_batch_size)
        train_loss_per_step.append(loss.item())

        outputs_for_metrics = (
            outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
        )
        results = process_outputs(args, outputs_for_metrics, label_names, sampling_rate)

        tasks_metrics = {}
        for task in tasks:
            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device,
            )
            tasks_metrics[task] = metrics
            metrics.compute(
                targets=metrics_targets[task],
                preds=results[task],
                reduce=is_dist_avail_and_initialized(),
            )
            for metric in metrics.metric_names():
                average_meters[f"{task}_{metric}"].update(
                    metrics.get_metric(name=metric), step_batch_size
                )
            metrics_merged[f"{task}"].add(metrics)

        if tensor_writer is not None and is_main_process():
            gstep = epoch * len(train_loader) + step
            tensor_writer.add_scalar("learning-rate/step", lr, gstep)
            tensor_writer.add_scalar("train-loss/step", loss.item(), gstep)
            for task in tasks:
                values = tasks_metrics[task].get_all_metrics()
                tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)

        if step % args.log_step == 0 and is_main_process():
            prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
            logger.info(prg_str)

    return train_loss_per_step, metrics_merged


def train_worker(args, device) -> str:
    """
    Launches the full training workflow including model setup, data loading, and training loop.

    This function initializes model, datasets, dataloaders, loss function, optimizer,
    and scheduler. It supports distributed training and automatic checkpointing for the best model.

    Args:
        args: Namespace or configuration object containing all training parameters.
        device (torch.device): The device used for training (CPU or GPU).

    Returns:
        str: Path to the saved model checkpoint with the best validation loss.
    """
    logger.set_logger("train")

    log_dir = logger.logdir()
    checkpoint_save_dir = get_safe_path(os.path.join(log_dir, "checkpoints"))
    tb_dir = get_safe_path(os.path.join(log_dir, "tensorboard"))

    tensor_writer = SummaryWriter(tb_dir) if args.use_tensorboard else None

    if is_main_process():
        with open(os.path.join(log_dir, f"run_tb_{get_time_str()}.sh"), "w") as f:
            f.write(f"tensorboard --logdir '{tb_dir}' --port 8080")
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)

    train_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="train",
    )

    val_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="val",
    )

    logger.info(f"train size: {len(train_dataset)}, val size:{len(val_dataset)}")

    train_sampler = (
        torch.utils.data.DistributedSampler(train_dataset)
        if is_dist_avail_and_initialized()
        else None
    )
    val_sampler = (
        torch.utils.data.DistributedSampler(val_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    persistent = args.workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
        persistent_workers=persistent,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=val_sampler,
        persistent_workers=persistent,
    )

    if args.steps > 0:
        args.epochs = math.ceil(args.steps / len(train_loader))
    args.steps = args.epochs * len(train_loader)
    logger.warning(f"`args.epochs` -> {args.epochs}, `args.steps` -> {args.steps}")

    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint,
            device=device,
            dist_mode=args.distributed,
            compile_mode=args.use_torch_compile,
            resume=True,
        )
        logger.info(f"Model loaded: {args.checkpoint}")
    else:
        checkpoint = None

    loss_fn = Config.get_loss(model_name=args.model_name)
    best_loss = (
        float("inf")
        if (checkpoint is None or "loss" not in checkpoint)
        else checkpoint["loss"]
    )
    loss_fn = loss_fn.to(device)

    model = create_model(
        model_name=args.model_name,
        in_channels=in_channels,
        in_samples=args.in_samples,
    )

    if checkpoint is not None and "model_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_dict"])
        logger.info(f"model.load_state_dict")

    if is_main_process():
        backup_path = get_safe_path(os.path.join(log_dir, "model_backup.py"))
        shutil.copy2(inspect.getfile(model.__class__), backup_path)
        logger.info(f"Model parameters: {count_parameters(model)}")

    model = model.to(device)

    optim_lower = args.optim.lower()
    if optim_lower == "adam":
        optimizer = torch.optim.Adam(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "adamw":
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "sgd":
        optimizer = torch.optim.SGD(
            [{"params": model.parameters(), "initial_lr": args.base_lr}],
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer:'{args.optim}'")
    if checkpoint is not None and "optimizer_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        logger.info(f"optimizer.load_state_dict")

    if args.use_lr_scheduler:
        if args.warmup_steps < 1:
            if args.warmup_steps > 0:
                args.warmup_steps = int(args.steps * args.warmup_steps)
            elif args.warmup_steps <= 0:
                args.warmup_steps = 1
            logger.info(f"`args.warmup_steps` will be set to `{args.warmup_steps}`")

        if args.down_steps < 1:
            if args.down_steps > 0:
                args.down_steps = int(args.steps * args.down_steps)
            elif args.down_steps <= 0:
                args.down_steps = args.steps - args.warmup_steps
            logger.info(f"`args.down_steps` will be set to `{args.down_steps}`")

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=args.warmup_steps,
            step_size_down=args.down_steps,
            mode=args.lr_scheduler_mode,
            gamma=args.base_lr ** ((args.steps * 2) ** -1),
            cycle_momentum=False,
            last_epoch=args.start_epoch * len(train_loader) - 1,
            verbose=False,
        )
    else:
        scheduler = None

    losses_dict = {
        n: []
        for n in ["train_loss_per_step", "train_loss_per_epoch", "val_loss_per_epoch"]
    }

    num_saved = 0
    epochs_since_improvement = 0

    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ckpt_path = None
    cost_time = datetime.timedelta()
    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        epoch_start_time = datetime.datetime.now()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch=epoch)

        train_losses, train_metrics_dict = train(
            args,
            model_tasks,
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_loader,
            epoch,
            device,
            tensor_writer,
        )
        train_loss = np.mean(train_losses)
        losses_dict["train_loss_per_step"].extend(train_losses)
        losses_dict["train_loss_per_epoch"].append(train_loss)

        val_loss, val_metrics_dict = validate(
            args, model_tasks, model, loss_fn, val_loader, epoch, device
        )
        losses_dict["val_loss_per_epoch"].append(val_loss)

        if is_main_process():
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(checkpoint_save_dir, f"model-{epoch}.pth")
                save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
                logger.info(f"Model saved: {ckpt_path}")
                num_saved += 1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                logger.info(f"Epochs since last improvement:{epochs_since_improvement}")

            if tensor_writer is not None:
                tensor_writer.add_scalars(
                    "train-val.loss/epoch",
                    {"train": train_loss, "val": val_loss},
                    epoch,
                )
                for task in model_tasks:
                    tensor_writer.add_scalars(
                        f"train.{task}.metrics/epoch",
                        train_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.metrics/epoch",
                        val_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.allvalues/epoch",
                        val_metrics_dict[task].to_dict(),
                        epoch,
                    )

            train_metrics_str = "* [Train Metrics]"
            val_metrics_str = "* [Val Metrics]"
            for task in model_tasks:
                train_metrics_str += f"[{task.upper()}]{train_metrics_dict[task]} "
                val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            logger.info(train_metrics_str)
            logger.info(val_metrics_str)

            if epochs_since_improvement > args.patience:
                logger.warning(f"\n* Stop training.")
                break

            epoch_end_time = datetime.datetime.now()
            epoch_cost_time = epoch_end_time - epoch_start_time
            cost_time += epoch_cost_time
            estimated_end_time = (
                (cost_time / (i + 1)) * 0.1 + epoch_cost_time * 0.9
            ) * (args.epochs - (i + 1)) + epoch_end_time
            logger.info(f"* Epoch cost time: {strftimedelta(epoch_cost_time)}")
            logger.info(
                f"* Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    if is_main_process():
        loss_save_dir = os.path.join(log_dir, "loss")
        if not os.path.exists(loss_save_dir):
            os.makedirs(loss_save_dir)
        for name, t in losses_dict.items():
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            np.save(os.path.join(loss_save_dir, f"{args.model_name}_{name}.npy"), t)

    if is_dist_avail_and_initialized():
        ckpt_path = broadcast_object(ckpt_path, src=0)

    return ckpt_path
