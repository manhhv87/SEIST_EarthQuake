from typing import Union
import os
import torch
import torch.distributed as dist
from config import Config
from utils import *
from .postprocess import process_outputs, ResultSaver
import json


def validate(
    args, tasks, model, loss_fn, val_loader, epoch, device, testing=False
) -> Union[float, dict]:
    """
    Evaluates the model's performance on a validation or testing dataset.

    This function evaluates the model on the provided validation or testing dataset. It computes
    the loss and metrics for each task, and saves the results if specified. The model's predictions
    are processed, and metrics are computed for each task. If `testing` is `True`, the results can
    be saved to a CSV file for further analysis.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration settings.
        tasks (list): List of tasks (e.g., classification, regression) to evaluate the model on.
        model (torch.nn.Module): The model to be evaluated.
        loss_fn (torch.nn.Module): The loss function used for computing the loss.
        val_loader (torch.utils.data.DataLoader): DataLoader for loading the validation or test dataset.
        epoch (int): The current epoch number.
        device (torch.device): The device (CPU/GPU) used for evaluation.
        testing (bool, optional): If True, the function performs testing instead of validation. Default is False.

    Returns:
        Union[float, dict]: The average loss and a dictionary of computed metrics for each task.

    Logs:
        - Loss for each validation/testing step.
        - Metrics for each task, including precision, recall, etc.
        - Progress of validation/testing steps if `is_main_process()`.

    Side effects:
        - Saves results to CSV if `testing=True` and `args.save_test_results=True`.
        - Logs validation/testing progress to the logger.
        - Updates the model's state for distributed training if applicable.
    """
    model.eval()

    # Get model configuration for labels, transformation functions, etc.
    model_labels, tgts_trans_for_loss, outs_trans_for_loss, outs_trans_for_res = (
        Config.get_model_config_(
            args.model_name,
            "labels",
            "targets_transform_for_loss",
            "outputs_transform_for_loss",
            "outputs_transform_for_results",
        )
    )

    # Initialize metrics and meters
    average_meters = {}
    metrics_merged = {}

    sampling_rate = val_loader.dataset.sampling_rate()
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

    # Set up progress meter
    progress = ProgressMeter(
        len(val_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test' if testing else 'Val'}: [{epoch}/{args.epochs}]",
    )

    # Initialize results saver if testing and saving results
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks)
    else:
        results_saver = None

    # Loop through validation or testing steps
    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(
            val_loader
        ):
            # Move data to device
            if isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
            else:
                x = x.to(device)

            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(device)

            # Forward pass
            outputs = model(x)

            # Loss computation
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

            # Batch size of current step
            step_batch_size = x.size(0)

            # Reduce loss for distributed training
            if is_dist_avail_and_initialized():
                loss = reduce_tensor(loss, "AVG")
                step_batch_size = torch.tensor(
                    step_batch_size, device=device, dtype=torch.int32
                )
                step_batch_size = reduce_tensor(step_batch_size)
                dist.barrier()
                step_batch_size = step_batch_size.item()

            # Update loss meter
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs for metrics
            outputs_for_metrics = (
                outs_trans_for_res(outputs)
                if outs_trans_for_res is not None
                else outputs
            )
            results = process_outputs(
                args, outputs_for_metrics, model_labels, sampling_rate
            )

            # Save test results if needed
            if results_saver is not None:
                if isinstance(meta_data_jsons, torch.Tensor):
                    meta_data_jsons = meta_data_jsons.detach().cpu().tolist()

                meta_data_dict = {k: [] for k in json.loads(meta_data_jsons[0]).keys()}
                for j in meta_data_jsons:
                    for k, v in json.loads(j).items():
                        meta_data_dict[k].append(v)
                results_saver.append(meta_data_dict, metrics_targets, results)

            # Compute metrics for each task
            for task in tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )

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

            # Log progress if necessary
            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(
                    batch_idx=step,
                    name=f"{args.model_name}_{'test' if testing else 'val'}",
                )
                logger.info(prg_str)

    # Save test results to CSV
    if results_saver is not None:
        results_save_path = get_safe_path(
            os.path.join(
                logger.logdir(), f"test_results_{val_loader.dataset.name()}.csv"
            )
        )
        results_saver.save_as_csv(results_save_path)

    # Return average loss and metrics
    loss_avg = average_meters["loss"].avg
    return loss_avg, metrics_merged
