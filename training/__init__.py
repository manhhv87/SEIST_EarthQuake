"""
Module containing core functions for training and testing the model.

Includes:
- train_worker: Manages the training process including data loading, optimization steps,
  weight updates, and logging of results.
- test_worker: Handles model evaluation on validation or test datasets,
  computing performance metrics and reporting outcomes.

These functions are defined in the submodules `train` and `test` respectively,
and are imported here for convenient access or re-export.
"""

from .train import train_worker
from .test import test_worker
