"""
Model package initialization module.

This module imports various seismic deep learning model architectures and
loss functions, as well as utility functions for model management such as
registration, creation, checkpoint saving, and loading.

Modules imported:
    - eqtransformer: EQTransformer model implementation.
    - phasenet: PhaseNet model implementation.
    - magnet: MAGNet model implementation.
    - baz_network: Baz network implementation.
    - distpt_network: DistPT network implementation.
    - ditingmotion: DiTingMotion model implementation.
    - seist: SEIST model implementation.
    - team: TEAM model implementation.

Loss functions imported:
    - CELoss: Cross-Entropy Loss.
    - MSELoss: Mean Squared Error Loss.
    - BCELoss: Binary Cross-Entropy Loss.
    - FocalLoss: Focal Loss for class imbalance.
    - BinaryFocalLoss: Binary version of Focal Loss.
    - CombinationLoss: Combination of multiple loss functions.
    - HuberLoss: Huber Loss for robust regression.
    - MousaviLoss: Custom loss function defined by Mousavi et al.

Factory utilities imported:
    - get_model_list: Retrieve the list of available models.
    - register_model: Decorator to register a new model.
    - create_model: Instantiate a model by name.
    - save_checkpoint: Save model checkpoint to file.
    - load_checkpoint: Load model checkpoint from file.
"""

from . import (
    eqtransformer,
    phasenet,
    magnet,
    baz_network,
    distpt_network,
    ditingmotion,
    seist,
    team
)
from .loss import CELoss, MSELoss, BCELoss, FocalLoss, BinaryFocalLoss, CombinationLoss, HuberLoss, MousaviLoss, NLLLoss
from ._factory import get_model_list, register_model, create_model, save_checkpoint, load_checkpoint
