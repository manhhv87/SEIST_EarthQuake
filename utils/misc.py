"""
This module provides various utility functions for setting seeds, handling distributed training,
calculating performance metrics, managing file paths, and other commonly used operations.

Functions:
- setup_seed: Sets the random seed for reproducibility across various libraries (PyTorch, NumPy, Python random).
- get_time_str: Returns the current time as a formatted string (YYYY-MM-DD_HH-MM-SS).
- strftimedelta: Converts a `timedelta` object into a human-readable string format ('Xh Ymin Zs').
- get_safe_path: Ensures that the provided file path does not exist, appending a tag if necessary.
- _setup_for_distributed: Configures printing for multi-process training setups, disabling it for non-master processes.
- is_dist_avail_and_initialized: Checks if distributed training is available and initialized.
- get_world_size: Returns the total number of processes in the distributed setup.
- get_rank: Returns the rank (ID) of the current process in the distributed setup.
- get_local_rank: Returns the local rank (ID) of the current process.
- is_main_process: Checks if the current process is the main process (rank 0).
- reduce_tensor: Performs an all-reduce operation across distributed processes.
- gather_tensors_to_list: Gathers tensors from all processes into a list.
- broadcast_object: Broadcasts an object from a specified source process.
- init_distributed_mode: Initializes the distributed environment for multi-GPU training.
"""

import datetime
import warnings
import os
import random
import re
from typing import Any, Dict, List, Literal, Union
import GPUtil
import numpy as np
import math
import torch
import torch.distributed as dist


def setup_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across various libraries (PyTorch, NumPy, Python random).

    Args:
        seed (int): The seed value to initialize the random generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_str() -> str:
    """Returns the current time as a formatted string in the format 'YYYY-MM-DD_HH-MM-SS'.

    Returns:
        str: A string representing the current timestamp.
    """
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return dtstr


def strftimedelta(td: datetime.timedelta) -> str:
    """Converts a `timedelta` object to a human-readable string format 'Xh Ymin Zs'.

    Args:
        td (datetime.timedelta): The timedelta object to be converted.

    Returns:
        str: A string representing the timedelta in hours, minutes, and seconds.
    """
    _seconds = int(td.seconds + td.microseconds // 1e6)
    hours = int(td.days * 24 + _seconds // 3600)
    minutes = int(_seconds % 3600 / 60)
    seconds = _seconds % 60
    deltastr = f"{hours}h {minutes}min {seconds}s"
    return deltastr


def get_safe_path(path: str, tag: str = "new") -> str:
    """Generates a unique file path that does not already exist by appending a tag to the base name.

    Args:
        path (str): The original file path.
        tag (str, optional): A tag to append to the path if it already exists. Defaults to "new".

    Returns:
        str: A safe, unique file path.
    """
    if is_main_process():
        d = os.path.split(path)[0]
        if not os.path.exists(d):
            os.makedirs(d)
    if os.path.exists(path):
        _tag = "_" + str(tag).replace(" ", "_")
        path = _tag.join(os.path.splitext(path))
        return get_safe_path(path, tag)
    else:
        return path


def _setup_for_distributed(is_master: bool) -> None:
    """Configures the print function for distributed training, ensuring that only the master process prints.

    Args:
        is_master (bool): A boolean indicating whether the current process is the master process.

    Reference:
        https://github.com/facebookresearch/detr/blob/main/util/misc.py
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Checks if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, otherwise False.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Returns the total number of processes in the distributed setup.

    Returns:
        int: The number of processes in the current distributed setup.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Returns the rank (ID) of the current process in the distributed setup.

    Returns:
        int: The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Returns the local rank (ID) of the current process in the distributed setup.

    Returns:
        int: The local rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0) in the distributed setup.

    Returns:
        bool: True if the current process is the main process, otherwise False.
    """
    return get_rank() == 0


def reduce_tensor(
    t: torch.Tensor, op: str = "SUM", barrier: bool = False
) -> torch.Tensor:
    """Performs an all-reduce operation across distributed processes.

    Args:
        t (torch.Tensor): The tensor to be reduced.
        op (str, optional): The operation to perform (e.g., 'SUM', 'AVG'). Defaults to 'SUM'.
        barrier (bool, optional): Whether to synchronize all processes after the reduction. Defaults to False.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    assert op in ["SUM", "AVG", "PRODUCT", "MIN", "MAX", "PREMUL_SUM"]
    _t = t.clone().detach()
    _op = getattr(dist.ReduceOp, op)
    dist.all_reduce(_t, op=_op)
    if barrier:
        dist.barrier()
    return _t


def gather_tensors_to_list(
    t: torch.Tensor, barrier: bool = False
) -> List[torch.Tensor]:
    """Gathers tensors from all processes into a list.

    Args:
        t (torch.Tensor): The tensor to be gathered.
        barrier (bool, optional): Whether to synchronize all processes after gathering. Defaults to False.

    Returns:
        List[torch.Tensor]: A list of tensors from all processes.
    """
    _t = t.clone().detach()
    _ts = [torch.zeros_like(_t) for _ in range(get_world_size())]
    dist.all_gather(_ts, _t)

    if barrier:
        dist.barrier()

    return _ts


def broadcast_object(obj: Any, src: int = 0, device: torch.device = None) -> Any:
    """Broadcasts an object from a source process to all other processes in the distributed setup.

    Args:
        obj (Any): The object to be broadcasted.
        src (int, optional): The source process. Defaults to 0.
        device (torch.device, optional): The device to use for the broadcast. Defaults to None.

    Returns:
        Any: The broadcasted object.
    """
    _obj = [obj]
    dist.broadcast_object_list(_obj, src=src, device=device)
    return _obj.pop()


def init_distributed_mode() -> bool:
    """Initializes the distributed environment for multi-GPU training using NCCL backend.

    Returns:
        bool: True if distributed training is initialized successfully, otherwise False.
    """
    # Arguments from environment
    required_args = set(["WORLD_SIZE", "RANK", "LOCAL_RANK"])
    if not required_args.issubset(os.environ):
        return False

    # For RTX 40xx Series
    for gpu in GPUtil.getGPUs():
        if re.findall(r"GEFORCE RTX", gpu.name, re.I):
            version = re.findall(r"\d+", gpu.name)
            if version and int(version[0]) > 4000:
                os.environ["NCCL_P2P_DISABLE"] = "1"
                os.environ["NCCL_IB_DISABLE"] = "1"
                os.environ["NCCL_DEBUG"] = "info"
                warnings.warn(
                    f"GPU ({gpu.name}) detected. 'NCCL_P2P' and 'NCCL_IB' are disabled."
                )
                break

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

    _setup_for_distributed(is_main_process())

    return True


# def adjust_learning_rate(args, optimizer, train_steps) -> float:
#     """
#     Adjust the learning rate based on the current training step, warm-up steps, and decay strategy.

#     This function adjusts the learning rate during training by either warming it up or applying a decay
#     strategy once the warm-up phase is over. The learning rate can be decayed using either an exponential
#     decay or a sine function, as specified by the arguments.

#     Args:
#         args (Namespace): A configuration object containing hyperparameters, including:
#             - lr (float): Initial learning rate.
#             - warmup_steps (int): The number of warm-up steps.
#             - decay_freq (int): Frequency at which to apply decay after warm-up.
#             - decay_op (str): Decay operation, either 'e' for exponential or 'sin' for sine decay.
#             - shrink_factor (float): Shrink factor used in exponential decay (only used when `decay_op` is 'e').
#         optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
#         train_steps (int): The current number of training steps.

#     Returns:
#         float: The new learning rate after adjustment.

#     Raises:
#         ValueError: If an unknown decay operation is provided in `decay_op`.
#     """
#     print("Now lr: ", optimizer.param_groups[0]["lr"])
#     if train_steps < args.warmup_steps:
#         percent = (train_steps +1 ) / args.warmup_steps
#         learning_rate = args.lr * percent
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = learning_rate
#         print("New learning rate:", learning_rate)
#     else:
#         if (train_steps - args.warmup_steps + 1) % args.decay_freq == 0:
#             if args.decay_op == "e":
#                 learning_rate = optimizer.param_groups[0]["lr"] ** args.shrink_factor
#             elif args.decay_op == "sin":
#                 learning_rate = np.sin(optimizer.param_groups[0]["lr"])
#             else:
#                 raise ValueError(f"`decay_op` must be 'e' or 'sin', got '{args.decay_op}'")
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = learning_rate
#             print("New learning rate:", learning_rate)
#     return optimizer.param_groups[0]["lr"]


def strfargs(args, configs) -> str:
    """
    Converts the arguments and configurations to a formatted string.

    This function takes two objects, `args` and `configs`, and converts their attributes
    into a human-readable string format. The function will include all arguments and configuration
    parameters, except those that are private (i.e., start and end with double underscores),
    callable, or methods.

    Args:
        args (Namespace): A configuration object containing argument values.
        configs (Namespace): A configuration object containing additional settings or configurations.

    Returns:
        str: A formatted string that represents the arguments and configurations.
    """

    string = ""
    string += "\nArguments:\n"
    for k, v in args.__dict__.items():
        string += f"{k}: {v}\n"
    string += "\nConfigs:\n"
    for k, v in configs.__dict__.items():
        if not (
            (k.startswith("__") and k.endswith("__"))
            or callable(v)
            or isinstance(v, (classmethod, staticmethod))
        ):
            string += f"{k}: {v}\n"
    return string


def count_parameters(module: torch.nn.Module) -> int:
    """
    Counts the total number of parameters in a PyTorch model.

    This function iterates over all parameters in a given PyTorch model and
    sums up their total number of elements.

    Args:
        module (torch.nn.Module): The PyTorch model whose parameters will be counted.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum([param.numel() for param in module.parameters()])


def cal_snr(
    data: np.ndarray, pat: int, window: int = 500, method: str = "power"
) -> float:
    """
    Estimates the Signal-to-Noise Ratio (SNR) for a given data based on phase arrival time.

    The function estimates the Signal-to-Noise Ratio (SNR) using either the power or standard deviation method.
    The phase arrival time (PAT) and window length determine the portion of the data considered for the SNR calculation.

    Args:
        data (np.ndarray): A 3-component dataset (C, L) where C is the number of channels and L is the length.
        pat (int): Phase arrival time, indicating where the signal of interest starts.
        window (int, optional): The length of the window used to calculate the SNR (in samples). Default is 500.
        method (str, optional): The method for calculating SNR. It can be either "power" or "std". Default is "power".

    Returns:
        float: The estimated SNR in decibels (dB).

    Raises:
        Exception: If an unsupported method is provided for SNR calculation.

    Modified from:
        https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/predictor.py
    """

    pat = int(pat)

    assert window < data.shape[-1] / 2, f"window = {window}, data.shape = {data.shape}"
    assert 0 < pat < data.shape[-1], f"pat = {pat}"

    if (pat + window) <= data.shape[-1]:
        if pat >= window:
            nw = data[:, pat - window : pat]
            sw = data[:, pat : pat + window]
        else:
            window = pat
            nw = data[:, pat - window : pat]
            sw = data[:, pat : pat + window]
    else:
        window = data.shape[-1] - pat
        nw = data[:, pat - window : pat]
        sw = data[:, pat : pat + window]

    if method == "power":
        snr = np.mean(sw**2) / (np.mean(nw**2) + 1e-6)
    elif method == "std":
        snr = np.std(sw) / (np.std(nw) + 1e-6)
    else:
        raise Exception(f"Unknown method: {method}")

    snr_db = round(10 * np.log10(snr), 2)

    return snr_db
