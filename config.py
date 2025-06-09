"""
This module defines the `Config` class for managing model configurations, loss functions,
input/output items, and evaluation metrics for various seismic models.

It includes functionality to:
- Define and store configuration settings for different models (e.g., PhaseNet, EQTransformer, etc.).
- Manage the available input/output items and their associated types and metrics.
- Retrieve loss functions and other configurations for specific models.
- Check and initialize configurations to ensure correctness.

The module imports necessary components from other libraries like PyTorch and custom loss functions
to support model configurations.

Classes:
    Config: A class that manages model configurations, input/output items, and evaluation metrics.

    Model Configurations:
        The model configurations are stored in the `Config.models` dictionary. Each model is defined by:
            - `loss`: Loss function used by the model.
            - `inputs`: The input data required for the model.
            - `labels`: Labels for training the model.
            - `eval`: Evaluation tasks to be performed during training.
            - `targets_transform_for_loss`: Transformations for target data before computing loss.
            - `outputs_transform_for_loss`: Transformations for output data before computing loss.
            - `outputs_transform_for_results`: Transformations for output data after generating results.

Module Functions:
    - `check_and_init()`: Checks and initializes configurations for models and input/output items.
    - `get_io_items(type=None)`: Retrieves available input/output items based on type.
    - `get_type(name)`: Returns the type of a given input/output item.
    - `get_num_classes(name)`: Retrieves the number of classes for a "onehot" item.
    - `get_model_config(model_name)`: Retrieves the configuration for a given model.
    - `get_model_config_(model_name, *attrs)`: Retrieves specific attributes from a model's configuration.
    - `get_num_inchannels(model_name)`: Retrieves the number of input channels for a model.
    - `get_metrics(item_name)`: Retrieves the metrics associated with a given input/output item.
    - `get_loss(model_name)`: Creates and returns an instance of the loss function for a model.
"""

from collections import defaultdict
import math
import re
from functools import partial
from typing import Any, Callable
import torch
from models import (
    BCELoss,
    BinaryFocalLoss,
    CELoss,
    CombinationLoss,
    FocalLoss,
    HuberLoss,
    MousaviLoss,
    MSELoss,
    NLLLoss,
    get_model_list,
)


class Config:
    """
    Configuration class for managing model configurations, available inputs/outputs,
    metrics, and loss functions.

    This class includes functionality to check and initialize model configurations,
    retrieve model-specific settings, and handle input-output items.

    Attributes:
        models (dict): A dictionary of model configurations, where keys are model names
                        and values are configuration dictionaries containing loss functions,
                        input data, labels, and other relevant information.
        _avl_metrics (tuple): A tuple of available metrics for evaluation.
        _avl_io_item_types (tuple): A tuple of available input/output item types.
        _avl_io_items (dict): A dictionary defining available input/output items and their
                               types, metrics, and other related details.
    """

    _model_conf_keys = (
        "loss",
        "labels",
        "eval",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )

    models = {
        # --------------------------------------------------------- PhaseNet
        "phasenet": {
            "loss": partial(CELoss, weight=[[1], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- EQTransformer
        "eqtransformer": {
            "loss": partial(BCELoss, weight=[[0.5], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- MagNet
        "magnet": {
            "loss": MousaviLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: x[:, 0].reshape(-1, 1),
        },
        # --------------------------------------------------------- BAZ Network
        "baz_network": {
            "loss": partial(CombinationLoss, losses=[MSELoss, MSELoss]),
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": lambda x: (
                (x * math.pi / 180).cos(),
                (x * math.pi / 180).sin(),
            ),
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: torch.atan2(x[1], x[0])
            * 180
            / math.pi,
        },
        # --------------------------------------------------------- dist-PT Network
        # # The model has not been reproduced due to no travel time data in the DiTing dataset.
        # #
        # "distpt_network": {
        #     "loss": CombinationLoss(
        #         losses=[MousaviLoss,MousaviLoss],
        #         losses_weights=[0.1,0.9]
        #     ),
        #     "inputs": ["z", "n", "e"],
        #     "labels": ["dis",""],
        #     "eval": ["dis",""],
        #     "targets_transform_for_loss": None,
        #     "outputs_transform_for_loss": None,
        #     "outputs_transform_for_results": lambda x,x1: x[:, 0].reshape(-1, 1),x1[:, 0].reshape(-1, 1),
        # },
        # --------------------------------------------------------- TEAM Network
        "team": {
            "loss": NLLLoss,
            "inputs": ["z", "n", "e"],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- DiTingMotion
        "ditingmotion": {
            "loss": partial(CombinationLoss, losses=[FocalLoss, FocalLoss]),
            "inputs": [["z", "dz"]],
            "labels": ["clr", "pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda xs: [x.softmax(-1) for x in xs],
        },
        # --------------------------------------------------------- SeisT (Detection, Picking)
        "seist_.*?_dpk.*": {
            "loss": partial(BCELoss, weight=[[0.5], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (P-motion-polarity)
        "seist_.*?_pmp": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Magnitude)
        "seist_.*?_emg": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Azimuth)
        "seist_.*?_baz": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Distance)
        "seist_.*?_dis": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["dis"],
            "eval": ["dis"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
    }

    #  ////////////////////////////////////////////////////////////////////////////// Metrics
    _avl_metrics = (
        "precision",
        "recall",
        "f1",
        "mean",
        "std",
        "mae",
        "mape",
        "r2",
    )

    #  ////////////////////////////////////////////////////////////////////////////// Available input and output items
    _avl_io_item_types = ("soft", "value", "onehot")

    _avl_io_items = {
        # -------------------------------------------------------------------------- Channel-Z
        "z": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Channel-N
        "n": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Channel-E
        "e": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(Z)
        "dz": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(N)
        "dn": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(E)
        "de": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- 1-P(p)-P(s)
        "non": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(d)
        "det": {"type": "soft", "metrics": ["precision", "recall", "f1"]},
        # -------------------------------------------------------------------------- P(p)
        "ppk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape"],
        },
        # -------------------------------------------------------------------------- P(s)
        "spk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape"],
        },
        # -------------------------------------------------------------------------- P(p+)
        "ppk+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(s+)
        "spk+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(d+)
        "det+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- Phase-P indices
        "ppks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "r2"]},
        # -------------------------------------------------------------------------- Phase-S indices
        "spks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "r2"]},
        # -------------------------------------------------------------------------- Event magnitude
        "emg": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]},
        # -------------------------------------------------------------------------- Station magnitude
        "smg": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]},
        # -------------------------------------------------------------------------- Back azimuth
        "baz": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]},
        # -------------------------------------------------------------------------- Distance
        "dis": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]},
        # -------------------------------------------------------------------------- P motion polarity
        "pmp": {
            "type": "onehot",
            "metrics": ["precision", "recall", "f1"],
            "num_classes": 2,
        },
        # -------------------------------------------------------------------------- Clarity
        "clr": {
            "type": "onehot",
            "metrics": ["precision", "recall", "f1"],
            "num_classes": 2,
        },
    }
    #  ////////////////////////////////////////////////////////////////////////////// (DO NOT modify the following methods)

    @classmethod
    def check_and_init(cls):
        """
        Initializes and checks the configurations of models and input/output items.

        This method performs the following checks:
        - Verifies that all models are correctly configured and match the registered models.
        - Ensures that all model configuration keys are present.
        - Validates that all input/output items are correctly defined with appropriate types
          and metrics.

        Raises:
            Exception: If any configuration is missing or incorrect.
            NotImplementedError: If an unknown label, input, task, or item type is found.
        """
        cls._type_to_ioitems = defaultdict(list)

        for k, v in cls._avl_io_items.items():
            cls._type_to_ioitems[v["type"]].append(k)

        # Check models
        useless_model_conf = list(cls.models)
        registered_models = get_model_list()
        for reg_model_name in registered_models:
            for re_name in cls.models:
                if re.findall(re_name, reg_model_name):
                    if re_name in useless_model_conf:
                        useless_model_conf.remove(re_name)

        if len(useless_model_conf) > 0:
            print(f"Useless configurations: {useless_model_conf}")

        # Check models' configuration
        for name, conf in cls.models.items():
            missing_keys = set(cls._model_conf_keys) - set(conf)
            if len(missing_keys) > 0:
                raise Exception(f"Model:'{name}'  Missing keys:{missing_keys}")
            expanded_labels = sum(
                [g if isinstance(g, (tuple, list)) else [g] for g in conf["labels"]], []
            )
            unknown_labels = set(expanded_labels) - set(cls._avl_io_items)
            if len(unknown_labels) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown labels:{unknown_labels}"
                )

            expanded_inputs = sum(
                [g if isinstance(g, (tuple, list)) else [g] for g in conf["inputs"]], []
            )
            unknown_inputs = set(expanded_inputs) - set(cls._avl_io_items)
            if len(unknown_inputs) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown inputs:{unknown_labels}"
                )

            unknown_tasks = set(conf["eval"]) - set(cls._avl_io_items)
            if len(unknown_tasks) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown tasks:{unknown_tasks}"
                )

        # Check io-items
        for k, v in cls._avl_io_items.items():
            if v["type"] not in cls._avl_io_item_types:
                raise NotImplementedError(f"Unknown item type: {v['type']}, item: {k}")

            unknown_metrics = set(v["metrics"]) - set(cls._avl_metrics)
            if len(unknown_metrics) > 0:
                raise NotImplementedError(
                    f"Unknown metrics:{unknown_metrics} , item: {k}"
                )

    @classmethod
    def get_io_items(cls, type: str = None) -> list:
        """
        Returns a list of available input/output items of a specified type.

        Args:
            type (str): The type of input/output items to retrieve. If None, returns all
                        available items.

        Returns:
            list: A list of available input/output items based on the specified type.

        Raises:
            ValueError: If the specified type is not one of the available item types.
        """
        if type is None:
            return list(cls._avl_io_items)
        else:
            return cls._type_to_ioitems[type]

    @classmethod
    def get_type(cls, name: str) -> list:
        """
        Retrieves the type of a given input/output item.

        Args:
            name (str): The name of the input/output item.

        Returns:
            list: The type of the item (e.g., "soft", "value", "onehot").

        Raises:
            ValueError: If the input/output item name does not exist.
        """
        return cls._avl_io_items[name]["type"]

    @classmethod
    def get_num_classes(cls, name: str) -> int:
        """
        Retrieves the number of classes for a given input/output item, if the item type is
        "onehot".

        Args:
            name (str): The name of the input/output item.

        Returns:
            int: The number of classes if the item type is "onehot".

        Raises:
            ValueError: If the input/output item name does not exist.
            Exception: If the item type is not "onehot".
        """
        if name not in cls._avl_io_items:
            raise ValueError(f"Name {name} not exists.")

        item_type = cls._avl_io_items[name]["type"]
        if item_type != "onehot":
            raise Exception(f"Type of item '{name}' is '{item_type}'.")

        num_classes = cls._avl_io_items[name]["num_classes"]

        return num_classes

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """
        Retrieves the configuration for a given model.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: A dictionary containing the model configuration.

        Raises:
            NotImplementedError: If the model name is not recognized or registered.
            Exception: If there is an issue with the model's configuration.
        """
        registered_models = get_model_list()

        if model_name not in registered_models:
            raise NotImplementedError(
                f"Unknown model:'{model_name}', registered: {registered_models}"
            )

        tgt_model_conf_keys = []
        for re_name in cls.models:
            if re.findall(re_name, model_name):
                tgt_model_conf_keys.append(re_name)
        if len(tgt_model_conf_keys) < 1:
            raise Exception(f"Missing configuration of model {model_name}")
        elif len(tgt_model_conf_keys) > 1:
            raise Exception(
                f"Model {model_name} matches multiple configuration items: {tgt_model_conf_keys}"
            )
        tgt_conf_key = tgt_model_conf_keys.pop()

        conf = cls.models[tgt_conf_key]

        return conf

    @classmethod
    def get_model_config_(cls, model_name: str, *attrs) -> Any:
        """
        Retrieves specific attributes of a model's configuration.

        Args:
            model_name (str): The name of the model.
            *attrs: The attributes to retrieve from the model's configuration.

        Returns:
            Any: The value(s) of the requested attribute(s).

        Raises:
            Exception: If the model or attribute is not recognized or if the attribute is
                       missing in the model's configuration.
        """
        model_conf = cls.get_model_config(model_name=model_name)
        attrs_conf = []
        for attr_name in attrs:
            if attr_name not in model_conf:
                raise Exception(
                    f"Unknown attribute:'{attr_name}', supported: {list(model_conf)}"
                )
            attrs_conf.append(model_conf[attr_name])
        if len(attrs_conf) == 1:
            conf = attrs_conf[0]
        else:
            conf = tuple(attrs_conf)
        return conf

    @classmethod
    def get_num_inchannels(cls, model_name: str) -> int:
        """
        Retrieves the number of input channels for a given model.

        Args:
            model_name (str): The name of the model.

        Returns:
            int: The number of input channels.

        Raises:
            Exception: If the number of input channels is incorrect or not found.
        """
        in_channels = 0
        inps = cls.get_model_config_(model_name, "inputs")
        for inp in inps:
            if isinstance(inp, (list, tuple)):
                if cls._avl_io_items[inp[0]]["type"] == "soft":
                    in_channels = len(inp)
                    break

        if in_channels < 1:
            raise Exception(
                f"Incorrect input channels. Model:{model_name} Inputs:{inps}"
            )
        return in_channels

    @classmethod
    def get_metrics(cls, item_name: str) -> list:
        """
        Retrieves the metrics associated with a given input/output item.

        Args:
            item_name (str): The name of the input/output item.

        Returns:
            list: A list of metrics associated with the item (e.g., "precision", "recall").

        Raises:
            Exception: If the item name is not found in the available input/output items.
        """
        if item_name not in cls._avl_io_items:
            raise Exception(
                f"Unknown item:'{item_name}', supported: {list(cls._avl_io_items)}"
            )
        metrics = cls._avl_io_items[item_name]["metrics"]
        return metrics

    @classmethod
    def get_loss(cls, model_name: str):
        """
        Creates and returns an instance of the loss function associated with the given model.

        Args:
            model_name (str): The name of the model.

        Returns:
            nn.Module: An instance of the loss function associated with the model.

        Raises:
            NotImplementedError: If the model name is not recognized or registered.
        """
        Loss = cls.get_model_config(model_name)["loss"]
        return Loss()


Config.check_and_init()
