"""
This module provides a registry system for dataset builder functions.
It allows datasets to be registered by name, listed, and instantiated dynamically.

Features:
- Register dataset creation functions with a decorator.
- Retrieve a list of all registered dataset names.
- Build dataset instances by name with optional parameters.

The registry prevents duplicate registrations and integrates with module-level __all__.

Functions:
    get_dataset_list(): Returns a list of all registered dataset names.
    register_dataset(func): Decorator to register a dataset builder function.
    build_dataset(dataset_name, **kwargs): Instantiate a registered dataset by name.

Usage example:

    @register_dataset
    def my_dataset(**kwargs):
        return MyDatasetClass(**kwargs)

    available = get_dataset_list()
    dataset = build_dataset("my_dataset", seed=42, mode="train")
"""

import sys
from typing import Callable


__all__ = ["get_dataset_list", "register_dataset", "build_dataset"]


_name_to_creators = {}


def get_dataset_list():
    """Return a list of all registered dataset names.

    Returns:
        list: A list of strings representing the names of all registered datasets.
    """

    return list(_name_to_creators)


def register_dataset(func: Callable) -> Callable:
    """Register a dataset creation function.

    This decorator registers a dataset builder function under its function name.
    It ensures no duplication and optionally appends the name to the module's __all__.

    Args:
        func (Callable): The dataset builder function to be registered.

    Returns:
        Callable: The same function passed in, after registration.

    Raises:
        Exception: If a dataset with the same name is already registered.
    """

    dataset_name = func.__name__

    if dataset_name in _name_to_creators:
        raise Exception(f"Dataset '{dataset_name}' already exists.")

    dataset_module = sys.modules[func.__module__]

    if (
        hasattr(dataset_module, "__all__")
        and dataset_name not in dataset_module.__all__
    ):
        dataset_module.__all__.append(dataset_name)

    _name_to_creators[dataset_name] = func

    return func


def build_dataset(dataset_name: str, **kwargs):
    """Build and return a dataset instance by name.

    This function calls the registered dataset builder function
    corresponding to the provided dataset name, passing any additional keyword arguments.

    Args:
        dataset_name (str): The name of the registered dataset to build.
        **kwargs: Arbitrary keyword arguments passed to the dataset builder function.

    Returns:
        Any: The constructed dataset object.

    Raises:
        ValueError: If the dataset name is not registered.
    """

    if dataset_name not in _name_to_creators:
        raise ValueError(f"Dataset '{dataset_name}' does not exist.")

    Dataset = _name_to_creators[dataset_name]

    dataset = Dataset(**kwargs)

    return dataset
