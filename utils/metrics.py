"""
Module for computing and tracking various evaluation metrics.

This module provides the `Metrics` class for computing common evaluation metrics such as 
Precision, Recall, F1-score, Mean, Standard Deviation (Std), Mean Absolute Error (MAE), 
Mean Absolute Percentage Error (MAPE), and R² (R-squared) for machine learning models. 
It supports both classification and regression tasks and includes functionality for 
distributed training.

Classes:
    Metrics: A class for tracking and computing evaluation metrics including Precision, 
             Recall, F1, Mean, Std, MAE, MAPE, and R².

Usage Example:
    metrics = Metrics(
        task="classification",
        metric_names=["precision", "recall", "f1", "mae"],
        sampling_rate=1000,
        time_threshold=0.2,
        num_samples=1000,
        device=torch.device("cuda")
    )

    # Compute metrics
    metrics.compute(targets, preds)

    # Get a specific metric
    precision = metrics.get_metric("precision")

    # Get all metrics
    all_metrics = metrics.get_all_metrics()
"""

import torch
import torch.distributed as dist
import numpy as np
import math
from .misc import reduce_tensor, gather_tensors_to_list
from typing import Tuple
import copy
from typing import List, Dict, Union


class Metrics:
    """Class for computing and tracking evaluation metrics for machine learning models.

    The `Metrics` class computes various evaluation metrics such as Precision, Recall, F1-score, 
    Mean, Standard Deviation, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and 
    R² (R-squared). It supports both regression and classification tasks, and can synchronize metrics 
    across multiple processes in distributed settings.

    Attributes:
        _epsilon (float): A small value used to avoid division by zero in some metrics.
        _avl_regr_keys (tuple): Keys related to regression metrics (e.g., MAE, MAPE).
        _avl_cmat_keys (tuple): Keys related to confusion matrix-based metrics (e.g., TP, FP).
        _avl_metrics (tuple): List of supported metric names (e.g., "precision", "recall", "f1").
        device (torch.device): The device on which computations will be performed (e.g., "cuda").
        _t_thres (int): Time threshold for phase-picking tasks, based on sampling rate.
        _task (str): The task type, e.g., "classification", "regression", "ppk".
        _metric_names (tuple): List of metric names to compute (e.g., "precision", "recall").
        _num_samples (int): Number of samples in the dataset.
        _data (dict): Dictionary to store intermediate results for computed metrics.
        _tgts (torch.Tensor): Tensor storing the true target values.
        _results (dict): Dictionary storing the final computed metric values.
        _modified (bool): Flag indicating if metrics have been modified and need updating.
    """

    _epsilon = 1e-6
    _avl_regr_keys = ("sum_res", "sum_squ_res", "sum_abs_res", "sum_abs_per_res")
    _avl_cmat_keys = ("tp", "predp", "possp")
    _avl_metrics = ("precision", "recall", "f1", "mean", "std", "mae", "mape", "r2")


    def __init__(
        self,
        task: str,
        metric_names: Union[list, tuple],
        sampling_rate: int,
        time_threshold: int,
        num_samples: int,
        device: torch.device,
    ) -> None:
        """Initializes the Metrics class for a given task and set of metrics.

        This constructor initializes an instance of the `Metrics` class, setting up the necessary 
        parameters for computing and storing metrics, such as task type, selected metrics, 
        sampling rate, and the device used for computations.

        Args:
            task (str): The task type (e.g., "classification", "regression").
            metric_names (Union[list, tuple]): List of metrics to compute (e.g., ["precision", "recall"]).
            sampling_rate (int): Sampling rate of the waveform.
            time_threshold (int): Threshold for phase-picking.
            num_samples (int): Number of samples in the dataset.
            device (torch.device): The device for computations (e.g., torch.device("cuda")).

        Raises:
            AssertionError: If an unsupported metric name is provided.

        Example:
            metrics = Metrics(
                task="classification",
                metric_names=["precision", "recall", "f1"],
                sampling_rate=1000,
                time_threshold=2,
                num_samples=5000,
                device=torch.device("cuda")
            )
            # Initializes a `Metrics` object for classification with precision, recall, and f1 metrics.
        """
        self.device = device

        self._t_thres = int(time_threshold * sampling_rate)

        self._task = task.lower()
        self._metric_names = tuple(n.lower() for n in metric_names)

        self._num_samples = num_samples

        unexpected_keys = set(self._metric_names) - set(self._avl_metrics)
        assert set(self._metric_names).issubset(
            self._avl_metrics
        ), f"Unexpected metrics:{unexpected_keys}"

        data_keys = self._metric_names
        if set(self._metric_names) & set(("precision", "recall", "f1")):
            data_keys += self._avl_cmat_keys
        if set(self._metric_names) & set(("mean", "std", "mae", "mape")):
            data_keys += self._avl_regr_keys

        self._data={
            k: torch.tensor(0, dtype=torch.float32, device=self.device)
            for k in data_keys
        }
        self._data["data_size"] = torch.tensor(0, dtype=torch.long, device=self.device)
        self._tgts: torch.Tensor = None
        self._results: Dict[str,float] = {}
        self._modified = True
        
    def synchronize_between_processes(self):
        """Synchronizes metrics across multiple processes during distributed training.

        This method ensures that the computed metrics are consistent across different processes 
        in a distributed environment. It performs reductions across tensors (e.g., sums) and 
        gathers the tensors from each process to ensure synchronization.

        The method uses `dist.barrier()` to synchronize processes, followed by reducing 
        tensors (e.g., summing metric values) across processes. It also gathers target tensors 
        (`_tgts`) if applicable.

        Example:
            metrics.synchronize_between_processes()
            # This will synchronize the metrics across different processes in a distributed setup.
        """
        dist.barrier()

        for k in self._data:
            self._data[k] = reduce_tensor(self._data[k])

        if isinstance(self._tgts, torch.Tensor):
            tgts_list = gather_tensors_to_list(self._tgts)
            self._tgts = torch.cat(tgts_list, dim=0)

        dist.barrier()

        self._modified = True

    def _order_phases(
        self, targets: torch.Tensor, preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Matches the order of predictions and true labels for phase-picking tasks.

        This method reorders the predicted phases to match the true phases. It computes a 
        distance matrix between the true and predicted phase values and assigns the closest 
        predicted phase to each true phase. This ensures that the predictions are correctly 
        aligned with the true labels for phase-based tasks (e.g., in time series or phase-based problems).

        Args:
            targets (torch.Tensor): True labels tensor, where each element represents the true phase.
            preds (torch.Tensor): Predictions tensor, where each element represents the predicted phase.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Ordered targets and predictions tensors, aligned with each other.

        Example:
            ordered_targets, ordered_preds = metrics._order_phases(targets, preds)
            # This will return the targets and predictions with matching orders.
        """
        num_phases = targets.size(-1)
        _targets = targets.clone().detach().cpu().numpy()
        _preds = preds.clone().detach().cpu().numpy()

        for i, (target_i, pred_i) in enumerate(zip(_targets, _preds)):
            orderd = np.zeros_like(pred_i)
            dmat = np.abs(
                target_i[:, np.newaxis].repeat(num_phases, axis=1)
                - pred_i[np.newaxis, :].repeat(num_phases, axis=0)
            )
            for _ in range(num_phases):
                ind = dmat.argmin()
                ito, ifr = ind // num_phases, ind % num_phases
                orderd[ito] = pred_i[ifr]
                dmat[ito, :] = int(1 / self._epsilon)
                dmat[:, ifr] = int(1 / self._epsilon)
            _preds[i] = orderd

        preds.copy_(torch.from_numpy(_preds))
        return targets, preds

    @torch.no_grad()
    def compute(
        self, targets: torch.Tensor, preds: torch.Tensor, reduce: bool = False
    ) -> None:
        """Computes the evaluation metrics based on the given targets and predictions.

        This method calculates various evaluation metrics such as precision, recall, 
        F1 score, mean absolute error (MAE), and others, based on the provided ground 
        truth labels (`targets`) and model predictions (`preds`). The method handles 
        different tasks like PPK (point prediction), SPK (sequence prediction), 
        and DET (detection) differently, applying specific formulas for each.

        Args:
            targets (torch.Tensor): Ground truth labels. Shape could be one of the following:
                - (N, L): For multi-class or multi-label classification tasks.
                - (N, Classes): For multi-class classification tasks.
                - (N, 2 * D): For bounding box tasks, where D is the dimension.
                - (N, 1): For single-value regression tasks.
            preds (torch.Tensor): Model predictions. Shape should match `targets`.
            reduce (bool, optional): Whether to perform metric synchronization across 
                processes in distributed training. Defaults to `False`.

        Raises:
            AssertionError: If the dimensions of `targets` and `preds` do not match, or if 
                the dimensions are incorrect for a given task.

        Example:
            metrics = Metrics(...)
            metrics.compute(targets, preds, reduce=True)
            # This will compute precision, recall, F1 score, etc., based on targets and preds.
        """
        assert targets.size(0) == preds.size(0), f"`{targets.size()}` != `{preds.size()}`"
        assert targets.dim() == 2, f"dim:{targets.dim()}, shape:{targets.size()}"

        self._data["data_size"] += targets.size(0)

        targets = targets.clone().detach().to(self.device)
        preds = preds.clone().detach().to(self.device)
        mask = 1.0

        if set(self._metric_names) & set(("precision", "recall", "f1")):
            if self._task in ["ppk", "spk"]:
                targets = targets.type(torch.long)
                preds = preds.type(torch.long)
                if targets.size(-1) > 1:
                    targets, preds = self._order_phases(targets, preds)

                preds_bin = (preds >= 0) & (preds < self._num_samples)
                targets_bin = (targets >= 0) & (targets < self._num_samples)
                
                ae = torch.abs(targets - preds)
                mask = tp_bin = preds_bin & targets_bin & (ae <= self._t_thres)

                self._data["tp"] = torch.sum(tp_bin)
                self._data["predp"] = torch.sum(preds_bin)
                self._data["possp"] = torch.sum(targets_bin)

            elif self._task in ["det"]:
                targets = targets.type(torch.long)
                preds = preds.type(torch.long)
                
                bs = targets.size(0)
                
                targets = targets.reshape(bs,-1,2)
                preds = preds.reshape(bs,-1,2)

                indices = torch.arange(self._num_samples,device=self.device).unsqueeze(0).unsqueeze(0)
                
                targets_bin = torch.sum((targets[:,:,:1] <= indices) & (indices <=targets[:,:,1:]),dim=-2)
                preds_bin = torch.sum((preds[:,:,:1] <= indices) & (indices <=preds[:,:,1:]),dim=-2)

                self._data["tp"] = torch.sum(
                    torch.round(torch.clip(targets_bin * preds_bin, 0, 1))
                )
                self._data["predp"] = torch.sum(
                    torch.round(torch.clip(preds_bin, 0, 1))
                )
                self._data["possp"] = torch.sum(
                    torch.round(torch.clip(targets_bin, 0, 1))
                )

            else:
                assert (
                    targets.size() == preds.size()
                ), f"`{targets.size()}` != `{preds.size()}`"
                assert targets.size(-1) > 1, f"The input must be one-hot."

                # Scatter is faster than fancy indexing.
                preds_indices = preds.topk(1).indices
                preds = preds.zero_().scatter_(dim=1,index=preds_indices,value=1)
                
                targets_indices = targets.topk(1).indices
                targets = targets.zero_().scatter_(dim=1,index=targets_indices,value=1)

                self._data["tp"] = torch.sum(targets * preds, dim=0)
                self._data["predp"] = torch.sum(preds, dim=0)
                self._data["possp"] = torch.sum(targets, dim=0)

        if set(self._metric_names) & set(("mean", "std", "mae", "mape", "r2")):
            res = targets - preds
            # BAZ
            if self._task in ["baz"]:
                res = torch.where(
                    res.abs() > 180, -torch.sign(res) * (360 - res.abs()), res
                )

            if "mean" in self._metric_names:
                self._data["sum_res"] = (res * mask).type(torch.float32).mean(-1).sum()

            if "std" in self._metric_names:
                self._data["sum_squ_res"] = (
                    torch.pow(res * mask, 2).type(torch.float32).mean(-1).sum()
                )

            if "mae" in self._metric_names:
                self._data["sum_abs_res"] = (
                    (res * mask).abs().type(torch.float32).mean(-1).sum()
                )

            if "mape" in self._metric_names:
                self._data["sum_abs_per_res"] = (
                    (res * mask / (targets + self._epsilon))
                    .abs()
                    .type(torch.float32)
                    .mean(-1)
                    .sum()
                )

            if "r2" in self._metric_names:
                self._tgts = targets
                if "sum_squ_res" not in self._data:
                    self._data["sum_squ_res"] = (
                        torch.pow(res * mask, 2).type(torch.float32).mean(-1).sum()
                    )

        if reduce:
            self.synchronize_between_processes()

        self._modified = True

    def add(self, b) -> None:
        """Adds the metrics from another `Metrics` object to the current one.

        This method allows you to combine the metrics from another `Metrics` object (`b`)
        into the current one. The metrics data fields of the two objects must match, otherwise,
        a `TypeError` will be raised. This method also concatenates the target tensors (`_tgts`)
        if they exist.

        Args:
            b (Metrics): Another `Metrics` object whose metrics will be added to the current object.

        Raises:
            TypeError: If `b` is not an instance of `Metrics` or if the data fields do not match.
        
        Example:
            metrics1.add(metrics2)
            # This will add the metrics from `metrics2` into `metrics1`.
        """
        if not type(self) == type(b):
            raise TypeError(f"Type of `b` must be `Metrics`, got `{type(b)}`")

        if (set(self._data) | set(b._data)) - (set(self._data) & set(b._data)):
            raise TypeError(
                f"Mismatched data fields: `{set(self._data)}` and `{set(b._data)}`"
            )

        for k in self._data:
            self._data[k] = self._data[k] + b._data[k]

        tgts_to_cat = list(
            filter(lambda x: isinstance(x, torch.Tensor), [self._tgts, b._tgts])
        )
        if tgts_to_cat:
            self._tgts = torch.cat(tgts_to_cat, dim=0)

        self._modified = True

    def __add__(a, b):
        """Adds two `Metrics` objects and returns a new `Metrics` object.

        This method enables the addition of two `Metrics` objects using the `+` operator.
        It returns a new `Metrics` object that contains the combined metrics of both `a` and `b`.
        The data fields of the two objects must match, otherwise, a `TypeError` will be raised.

        Args:
            b (Metrics): Another `Metrics` object to add to the current object.

        Returns:
            Metrics: A new `Metrics` object with the combined metrics.

        Raises:
            TypeError: If `b` is not an instance of `Metrics`, or if the data fields do not match.

        Example:
            metrics3 = metrics1 + metrics2
            # This will create a new `Metrics` object containing the sum of the metrics of `metrics1` and `metrics2`.
        """
        if not type(a) == type(b):
            raise TypeError(
                f"Unsupported operand type(s) for `+`: `{type(a)}` and `{type(b)}`"
            )

        if (set(a._data) | set(b._data)) - (set(a._data) & set(b._data)):
            raise TypeError(
                f"Mismatched data fields: `{set(a._data)}` and `{set(b._data)}`"
            )

        c = copy.deepcopy(a)
        for k in c._data:
            c._data[k] = a._data[k] + b._data[k]

        tgts_to_cat = list(
            filter(lambda x: isinstance(x, torch.Tensor), [a._tgts, b._tgts])
        )
        if tgts_to_cat:
            c._tgts = torch.cat(tgts_to_cat, dim=0)
        c._modified = True

        return c

    def _update_metric(self, key: str) -> torch.Tensor:
        """Computes and updates a specific metric.

        This method computes and updates a specified metric (e.g., "precision", "recall", "f1") 
        using the data stored in the `Metrics` object. The metric is calculated based on the 
        current values of true positives (tp), predicted positives (predp), and other 
        required variables. It returns the updated value of the metric as a `torch.Tensor`.

        Args:
            key (str): The metric to update (e.g., "precision", "recall", "f1", etc.).

        Returns:
            torch.Tensor: The updated value of the specified metric.

        Raises:
            ValueError: If the provided `key` is not a recognized metric name.

        Example:
            precision = metrics._update_metric("precision")
            # This will compute and return the updated precision value as a tensor.
        """
        if key == "precision":
            v = self._data["precision"] = (
                self._data["tp"] / (self._data["predp"] + self._epsilon)
            ).mean()
        elif key == "recall":
            v = self._data["recall"] = (
                self._data["tp"] / (self._data["possp"] + self._epsilon)
            ).mean()
        elif key == "f1":
            pr = self._data["tp"] / (self._data["predp"] + self._epsilon)
            re = self._data["tp"] / (self._data["possp"] + self._epsilon)
            v = self._data["f1"] = (2 * pr * re / (pr + re + self._epsilon)).mean()
        elif key == "mean":
            v = self._data["mean"] = self._data["sum_res"] / self._data["data_size"]
        elif key == "std":
            v = self._data["std"] = torch.sqrt(
                self._data["sum_squ_res"] / self._data["data_size"]
            )
        elif key == "mae":
            v = self._data["mae"] = self._data["sum_abs_res"] / self._data["data_size"]
        elif key == "mape":
            v = self._data["mape"] = (
                self._data["sum_abs_per_res"] / self._data["data_size"]
            )
        elif key == "r2":
            t = self._tgts - self._tgts.mean()
            # BAZ
            if self._task in ["baz"]:
                t = torch.where(t.abs() > 180, -torch.sign(t) * (360 - t.abs()), t)
            v = 1 - (
                self._data["sum_squ_res"]
                / (torch.pow(t, 2).mean(-1).sum() + self._epsilon)
            )
        else:
            raise ValueError(f"Unexpected key name: '{key}'")

        return v

    def _update_all_metrics(self) -> dict:
        """Updates all metrics and stores them in the results dictionary.

        This method updates the values for all metrics that have been computed so far. It will 
        call the corresponding update function for each metric and store the results in the `_results` 
        dictionary. If any metric has been modified or if the metrics have not been updated yet, 
        the method will recalculate all metrics.

        Returns:
            dict: A dictionary containing the names and values of all the updated metrics. 
                Each key is the metric name (e.g., 'precision', 'recall') and each value is the 
                computed metric value (float).

        Example:
            metrics = Metrics(...)
            all_metrics = metrics._update_all_metrics()
            # Output: {'precision': 0.8473, 'recall': 0.7832, 'f1': 0.8124, ...}
        """
        if self._modified or len(self._results)==0:
            self._results = {
                k: self._update_metric(k).item() for k in self._metric_names
            }
            self._modified = False
        return self._results

    def get_metric(self, name: str) -> float:
        """Returns the computed value for a specific metric.

        This method retrieves the computed value of a single metric from the results. It ensures 
        that all metrics are updated before fetching the value of the requested metric.

        Args:
            name (str): The name of the metric to retrieve (e.g., "precision", "recall").

        Returns:
            float: The computed value of the requested metric.

        Example:
            metrics = Metrics(...)
            precision_value = metrics.get_metric("precision")
            # Output: 0.8473
        """
        self._update_all_metrics()
        return self._results[name]

    def get_metrics(self, names: List[str]) -> Dict[str, float]:
        """Returns the computed values for a list of specific metrics.

        This method retrieves the computed values of the metrics specified in the 
        `names` argument. It updates all the metrics first if they are not updated already, 
        and then returns the requested metrics as a dictionary.

        Args:
            names (List[str]): A list of metric names (e.g., "precision", "recall", "f1") 
                                for which the values are to be retrieved.

        Returns:
            dict: A dictionary where the keys are the requested metric names and the values 
                are the corresponding computed metric values.

        Example:
            metrics = metrics.get_metrics(["precision", "f1"])
            print(metrics["precision"])  # Output: 0.85
            print(metrics["f1"])         # Output: 0.88
        """
        self._update_all_metrics()
        metrics_dict = {}
        for name in names:
            name_lower = name.lower()
            if name_lower in self._avl_metrics:
                metrics_dict[name] = self.get_metric(name_lower)
        return metrics_dict

    def metric_names(self) -> List[str]:
        """Returns the names of all the computed metrics.

        This method returns a list of all metric names that have been computed and 
        are available for retrieval.

        Returns:
            list: A list of strings representing the names of all available metrics 
                (e.g., ["precision", "recall", "f1", "mean", "mae"]).

        Example:
            metrics_list = metrics.metric_names()
            print(metrics_list)  # Output: ["precision", "recall", "f1", "mean", "mae"]
        """
        return list(self._metric_names)

    def get_all_metrics(self) -> Dict[str, float]:
        """Returns all the computed metrics as a dictionary.

        This method updates and returns all the computed metrics, with metric names as keys 
        and the corresponding computed values as float.

        Returns:
            dict: A dictionary where the keys are metric names and the values are 
                the corresponding computed metric values.

        Example:
            all_metrics = metrics.get_all_metrics()
            print(all_metrics)  # Output: {'precision': 0.85, 'recall': 0.78, 'f1': 0.81, ...}
        """
        return self._update_all_metrics()

    def __repr__(self) -> str:
        """Returns a string representation of the computed metrics.

        This method generates a string that represents all the computed metrics and their 
        corresponding values in a readable format, useful for displaying metrics in logs or 
        print statements.

        Returns:
            str: A string displaying all computed metrics and their values in the format 
                `METRIC_NAME value`, where `value` is a floating point number rounded to 4 decimal places.

        Example:
            metrics = Metrics(...)
            print(metrics)  # Output: PRECISION 0.8473  RECALL 0.7832  F1 0.8124  ...
        """
        entries = [
            f"{k.upper()} {v:6.4f}" for k, v in self._update_all_metrics().items()
        ]
        string = "  ".join(entries)
        return string

    def to_dict(self) -> dict:
        """Converts all computed metrics into a dictionary.

        This method converts the computed metrics into a dictionary format where each key 
        represents the metric name and the corresponding value is the computed metric value. 
        This is useful for serializing the metrics or for exporting them to a file.

        Returns:
            dict: A dictionary where the keys are metric names (strings) and the values are 
                the corresponding computed metric values, which can be either scalar or 
                lists of values depending on the metric type.

        Example:
            metrics = Metrics(...)
            metrics_dict = metrics.to_dict()
            # Output: {'precision': 0.8473, 'recall': 0.7832, 'f1': 0.8124, ...}
        """
        self._update_all_metrics()
        metrics_dict = {}
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                v = v.item() if v.dim() == 0 else v.tolist()

            if isinstance(v, (list, tuple, np.ndarray)):
                for i, vi in enumerate(v):
                    if isinstance(vi, torch.Tensor):
                        vi = vi.item()
                    metrics_dict[f"{k}.{i}"] = vi
            else:
                metrics_dict[k] = v
        return metrics_dict
