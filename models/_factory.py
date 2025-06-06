"""
Model management utilities module.

This module provides functions to register, create, save, and load deep learning
models. It maintains a registry of model creator functions and handles checkpointing
including support for distributed training and torch.compile modes.

Exports:
    get_model_list: Get the list of registered model names.
    register_model: Decorator to register a new model creation function.
    create_model: Instantiate a model by name with given arguments.
    save_checkpoint: Save training checkpoint including model and optimizer states.
    load_checkpoint: Load training checkpoint with support for device mapping and warnings on mode mismatches.
"""

import sys
import os
import warnings
import torch
from typing import Callable


__all__ = [
    "get_model_list",
    "register_model",
    "create_model",
    "save_checkpoint",
    "load_checkpoint",
]


_name_to_creators = {}


def get_model_list():
    """
    Get the list of registered model names.

    Returns:
        list[str]: List of registered model names.
    """
    return list(_name_to_creators)


def register_model(func: Callable) -> Callable:
    """
    Decorator to register a model creation function.

    Adds the decorated function to the internal registry of model creators.
    Also appends the model name to the module's __all__ list if it exists.

    Args:
        func (Callable): Model creation function to register.

    Returns:
        Callable: The original function, unmodified.

    Raises:
        Exception: If a model with the same name is already registered.
    """

    model_name = func.__name__

    if model_name in _name_to_creators:
        raise Exception(f"Model '{model_name}' already exists.")

    model_module = sys.modules[func.__module__]

    if hasattr(model_module, "__all__") and model_name not in model_module.__all__:
        model_module.__all__.append(model_name)

    _name_to_creators[model_name] = func

    return func


def create_model(model_name: str, **kwargs):
    """
    Create and return an instance of a registered model.

    Args:
        model_name (str): Name of the model to instantiate.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        object: Instantiated model object.

    Raises:
        ValueError: If the requested model_name is not registered.
    """

    if model_name not in _name_to_creators:
        avl_model_names = get_model_list()
        raise ValueError(f"Model '{model_name}' does not exist. \nAvailable: {avl_model_names}")

    Model = _name_to_creators[model_name]

    model = Model(**kwargs)

    return model


def save_checkpoint(
    save_path: str, epoch: int, model, optimizer, best_loss: float
) -> None:
    """
    Save a training checkpoint to disk.

    Handles models wrapped with DistributedDataParallel or torch.compile by
    saving the underlying module state dict. Stores epoch, optimizer state,
    best loss, and flags indicating usage of DDP and torch.compile.

    Args:
        save_path (str): Path to save the checkpoint file.
        epoch (int): Current training epoch number.
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer whose state to save.
        best_loss (float): Best loss value observed so far.
    """
    
    if hasattr(model, "module"):
        model_without_ddp = model.module
        use_ddp = True
    else:
        model_without_ddp = model
        use_ddp = False

    if hasattr(model, "_orig_mod"):
        model_without_compile = model_without_ddp._orig_mod
        use_compile = True
    else:
        model_without_compile = model_without_ddp
        use_compile = False
        
    torch.save(
        {
            "epoch": epoch,
            "optimizer_dict": optimizer.state_dict(),
            "model_dict": model_without_compile.state_dict(),
            "loss": best_loss,
            "use_compile": use_compile,
            "use_ddp": use_ddp,
        },
        save_path,
    )


def load_checkpoint(
    save_path: str,
    device: torch.device,
    dist_mode=False,
    compile_mode=False,
    resume=False,
):
    """
    Load a training checkpoint from disk.

    Cleans up state dict keys by removing DDP or torch.compile wrappers.
    Optionally warns if current distributed or compile modes differ from those
    used during training (useful for reproducibility awareness).

    Args:
        save_path (str): Path of the checkpoint file to load.
        device (torch.device): Device to map the loaded tensors onto.
        dist_mode (bool, optional): Whether distributed mode is currently enabled. Defaults to False.
        compile_mode (bool, optional): Whether torch.compile mode is currently enabled. Defaults to False.
        resume (bool, optional): Whether this loading is to resume training (enables warnings). Defaults to False.

    Returns:
        dict: Loaded checkpoint dictionary containing keys like 'model_dict', 'optimizer_dict', etc.
    """

    checkpoint = torch.load(save_path, map_location=device)
        
    if "model_dict" not in checkpoint:
        checkpoint = {"model_dict":checkpoint}

    checkpoint["model_dict"] = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in checkpoint["model_dict"].items()
    }

    if resume:
        used_ddp = checkpoint.get("use_ddp", False)
        used_compile = checkpoint.get("use_compile", False)
        if used_ddp != dist_mode:
            msg = (
                f"The model was trained {'without' if not used_ddp else 'with'} using distributed mode, "
                f"but distributed mode is {'enabled' if dist_mode else 'disabled'} now, "
                f"which may lead to unreproducible results."
            )
            warnings.warn(msg)
        if used_compile != compile_mode:
            msg = (
                f"The model was trained {'without' if not used_compile else 'with'} using `torch.compile` (triton random), "
                f"but the argument `use_torch_compile` is `{compile_mode}`, which may lead to unreproducible results."
            )
            warnings.warn(msg)

    return checkpoint
