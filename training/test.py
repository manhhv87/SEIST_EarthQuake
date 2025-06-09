import os
import torch
from config import Config
from models import create_model, load_checkpoint
from utils import *
from .preprocess import SeismicDataset
from .validate import validate


def test_worker(args, device) -> float:
    """
    Performs the testing process for a given model, loading the necessary dataset,
    applying the model, and evaluating its performance on the test set.

    This function loads the specified model from a checkpoint, prepares the test dataset,
    and calculates the test loss and metrics for each task. The evaluation is performed using
    distributed training when available. The results are logged and the final test loss is returned.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration settings.
        device (torch.device): The device (CPU/GPU) on which the model and data will be processed.

    Returns:
        float: The test loss value after evaluating the model on the test dataset.

    Raises:
        ValueError: If the checkpoint argument is not provided or is invalid.

    Logs:
        Logs the test progress, including:
            - Model loading status.
            - Test metrics for each task.
            - The test loss after evaluation.
    """

    # Log
    logger.set_logger("test")

    # Data loader setup
    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)
    test_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="test",
    )

    test_sampler = (
        torch.utils.data.DistributedSampler(test_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=test_sampler,
    )

    # Load checkpoint
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, device=device)
        logger.info(f"Model loaded: {args.checkpoint}")
    else:
        raise ValueError("checkpoint is None.")

    # Loss function
    loss_fn = Config.get_loss(model_name=args.model_name)
    loss_fn = loss_fn.to(device)

    # Model setup
    model = create_model(
        model_name=args.model_name,
        in_channels=in_channels,
        in_samples=args.in_samples,
    )
    if checkpoint is not None and "model_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_dict"])
        logger.info(f"model.load_state_dict")

    if is_main_process():
        logger.info(f"Model parameters: {count_parameters(model)}")

    model = model.to(device)

    # Distributed setup
    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Evaluate the model
    test_loss, test_metrics_dict = validate(
        args, model_tasks, model, loss_fn, test_loader, 0, device, testing=True
    )

    # Log test metrics
    if is_main_process():
        test_metrics_str = "* "
        for task in model_tasks:
            test_metrics_str += f"[{task.upper()}]{test_metrics_dict[task]} "
        logger.info(test_metrics_str)

    return test_loss
