"""
Module for tracking and displaying progress and averaging metrics during model training.

This module contains classes for tracking and visualizing metrics during training or evaluation, 
such as loss and accuracy. It includes functionality for averaging values over time and 
displaying progress with the number of completed batches.

Classes:
    - AverageMeter: A class to compute and store the average, sum, and count of a metric.
    - ProgressMeter: A class to display the progress of training or evaluation, including the current 
      batch number and associated metrics.

Example Usage:
    # Example of using AverageMeter
    avg_meter = AverageMeter("Loss")
    avg_meter.update(0.5, 1)  # Update with loss value
    print(str(avg_meter))  # Output: Loss 0.500000000000 (0.500000000000)

    # Example of using ProgressMeter
    meters = [AverageMeter("Loss"), AverageMeter("Accuracy")]
    progress_meter = ProgressMeter(100, meters)
    progress_meter.update(0.4, 0.9)  # Update with current values
    print(progress_meter.get_str(10, "Epoch 1"))
"""

import datetime

class AverageMeter(object):
    """A class to compute and store the average, sum, and count of a metric.

    This class is commonly used to track metrics like loss or accuracy during 
    training or evaluation. It stores the latest value, the running average, 
    the sum, and the count of updates made to the metric.

    Attributes:
        name (str): The name of the metric (e.g., "Loss", "Accuracy").
        fmt (str): The format specifier for displaying the values.
        val (float): The latest value added to the metric.
        avg (float): The running average of the metric.
        sum (float): The sum of all values added to the metric.
        count (int): The total number of updates made to the metric.
    """

    def __init__(self, name, fmt=":012f"):
        """Initializes the AverageMeter with a name and optional format.

        Args:
            name (str): The name of the metric (e.g., "Loss", "Accuracy").
            fmt (str, optional): The format specifier for displaying the values. Defaults to ":012f".
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all internal values to 0.

        This method is used to clear the previous statistics, starting fresh 
        for the next round of updates.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the metric with a new value and count.

        This method updates the running sum, count, and average of the metric.

        Args:
            val (float): The new value of the metric.
            n (int, optional): The number of samples represented by `val`. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a formatted string representation of the metric.

        This method returns a string representation of the latest value and 
        the running average of the metric.

        Returns:
            str: The formatted string displaying the metric's value and average.
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """A class to track and display the progress of training or evaluation.

    This class is used to display information about the progress of training, 
    including the current batch number, metrics, and the prefix (e.g., epoch number).

    Attributes:
        batch_fmtstr (str): The formatted string for displaying the batch number and total number of batches.
        meters (list): A list of AverageMeter objects to display metrics.
        prefix (str): A string to prefix the progress information (e.g., "Epoch 1").
    """

    def __init__(self, num_batches, meters, prefix=""):
        """Initializes the ProgressMeter with the total number of batches and the metrics to display.

        Args:
            num_batches (int): The total number of batches in the current epoch or training session.
            meters (list): A list of AverageMeter objects for the metrics to display.
            prefix (str, optional): A string to prefix the progress information (e.g., "Epoch 1").
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def get_str(self, batch_idx, name):
        """Returns a formatted string with the current progress information.

        This method returns a string that includes the batch index, the metric names, 
        and their current values.

        Args:
            batch_idx (int): The current batch index.
            name (str): The name of the progress, such as "Epoch 1".

        Returns:
            str: The formatted string displaying the current progress and metrics.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch_idx) + name]
        entries += [str(meter) for meter in self.meters]
        string = "  ".join(entries)
        return string

    def set_meters(self, meters):
        """Sets the meters (metrics) for the progress tracker.

        Args:
            meters (list): A list of AverageMeter objects representing the metrics to track.
        """
        self.meters = meters

    def _get_batch_fmtstr(self, num_batches):
        """Generates a formatted string for displaying the batch number and total batches.

        Args:
            num_batches (int): The total number of batches.

        Returns:
            str: The formatted string for displaying the batch number and total batches.
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
