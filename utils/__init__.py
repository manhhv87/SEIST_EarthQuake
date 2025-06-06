"""
This module provides a collection of utility functions and classes used for model evaluation, 
metric computation, logging, visualization, and distributed training. 

It includes tools for:
- Managing and tracking metrics (e.g., AverageMeter, ProgressMeter)
- Logging system (e.g., logger)
- Visualization (e.g., vis_waves_preds_targets, vis_phase_picking)
- Miscellaneous utilities for random seed setup, distributed training, file management, and more.

Modules and Functions:
-----------------------
- **AverageMeter**: Class for tracking the average value of a variable over time.
- **ProgressMeter**: Class for displaying progress of training/evaluation.
- **Metrics**: Class for computing and storing metrics for model evaluation.
- **logger**: Global logger for logging information during evaluation and training.
- **vis_waves_preds_targets**: Visualization function for comparing predicted and target waveform signals.
- **vis_phase_picking**: Visualization function for phase picking (used in signal processing).
- **setup_seed**: Function for setting the random seed for reproducibility.
- **get_time_str**: Function to return the current time as a formatted string.
- **strftimedelta**: Function to format a timedelta object into a string.
- **get_safe_path**: Function to ensure a safe file path, avoiding overwriting existing files.
- **is_dist_avail_and_initialized**: Checks if distributed training is available and initialized.
- **get_world_size**: Function to get the world size (total number of distributed processes).
- **get_rank**: Function to get the rank of the current process in a distributed setup.
- **get_local_rank**: Function to get the local rank of the current process.
- **is_main_process**: Function to check if the current process is the main process.
- **reduce_tensor**: Function to reduce a tensor across multiple processes (used in distributed training).
- **gather_tensors_to_list**: Gathers tensors from all processes into a list.
- **broadcast_object**: Broadcasts an object from the main process to all other processes in distributed training.
- **init_distributed_mode**: Initializes the distributed training environment.
- **strfargs**: Function to convert arguments into a formatted string.
- **count_parameters**: Function to count the number of parameters in a model.
- **cal_snr**: Function to calculate the Signal-to-Noise Ratio (SNR) of a signal.

This module is primarily used for model evaluation, visualization, distributed computing, and logging. It can be integrated into model training and evaluation pipelines to facilitate monitoring, visualization, and debugging.
"""


from .meters import AverageMeter, ProgressMeter
from .metrics import Metrics
from .logger import logger
from .visualization import vis_waves_preds_targets,vis_phase_picking
from .misc import (
    setup_seed,
    get_time_str,
    strftimedelta,
    get_safe_path,
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    reduce_tensor,
    gather_tensors_to_list,
    broadcast_object,
    init_distributed_mode,
    strfargs,
    count_parameters,
    cal_snr,
)
