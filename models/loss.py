import torch.nn as nn
import torch
from torch.nn import HuberLoss
from typing import Tuple


class CELoss(nn.Module):
    """
    Cross-Entropy Loss for multi-class classification with optional per-class weighting.

    Args:
        weight (list or torch.Tensor, optional): Weight for each class. If None, defaults to 1.0 (no weighting).

    Forward Input:
        preds (Tensor): Predicted probabilities or scores, shape (N, C, L) or (N, Classes).
        targets (Tensor): One-hot encoded target labels, same shape as preds.

    Forward Output:
        Tensor: Scalar loss value.
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        loss = -targets * torch.log(preds + self._epsilon)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss for binary classification tasks.

    Args:
        weight (float or list or torch.Tensor, optional): Weight applied to the loss. Defaults to 1.0.

    Forward Input:
        preds (Tensor): Predicted probabilities, shape (N, C, L).
        targets (Tensor): Binary ground truth labels, same shape as preds.

    Forward Output:
        Tensor: Scalar loss value.
    """

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weight:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification to address class imbalance.

    Args:
        gamma (float): Focusing parameter that reduces loss for well-classified examples. Default is 2.
        weight (list or torch.Tensor, optional): Weight per class. Defaults to 1.0 if None.
        has_softmax (bool): Whether to apply softmax to preds before loss calculation. Default is True.

    Forward Input:
        preds (Tensor): Predicted logits or probabilities, shape (N, C, L) or (N, Classes).
        targets (Tensor): One-hot encoded targets, same shape as preds.

    Forward Output:
        Tensor: Scalar loss value.
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, weight=None, has_softmax=True):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

        self.has_softmax = has_softmax

    def forward(self, preds, targets):
        if self.has_softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)

        loss = -targets * torch.log(preds + self._epsilon)
        loss *= torch.pow((1 - preds), self.gamma)
        loss *= self.weight
        loss = loss.sum(1).mean()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for binary classification with sigmoid outputs.

    Note:
        Input `preds` must be output of sigmoid activation.

    Args:
        gamma (float): Focusing parameter. Default is 2.
        alpha (float): Balancing factor between classes. Default is 1.
        weight (float or list or torch.Tensor, optional): Weight applied to the loss. Defaults to 1.0.

    Forward Input:
        preds (Tensor): Sigmoid probabilities, shape (N, C, L).
        targets (Tensor): Binary ground truth, same shape as preds.

    Forward Output:
        Tensor: Scalar loss value.
    """

    _epsilon = 1e-6

    def __init__(self, gamma=2, alpha=1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        loss = -(
            self.alpha
            * torch.pow((1 - preds), self.gamma)
            * targets
            * torch.log(preds + self._epsilon)
            + (1 - self.alpha)
            * torch.pow(preds, self.gamma)
            * (1 - targets)
            * torch.log(1 - preds + self._epsilon)
        )
        loss *= self.weight
        loss = loss.mean()
        return loss


class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) Loss.

    Args:
        weight (float or list or torch.Tensor, optional): Weight applied to the loss. Defaults to 1.0.

    Forward Input:
        preds (Tensor): Predicted values, shape (N, C, L).
        targets (Tensor): Ground truth values, same shape as preds.

    Forward Output:
        Tensor: Scalar loss value.
    """

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            print(f"[{self._get_name()}] Loss Weights:", weight)
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        loss = (preds - targets) ** 2
        loss *= self.weight
        loss = loss.mean()
        return loss


class CombinationLoss(nn.Module):
    """
    Combination loss module for multi-task learning that aggregates multiple losses.

    Args:
        losses (list): List of loss classes (not instances). Use `functools.partial` to pass arguments if needed.
        losses_weights (list, optional): List of float weights for each loss. If None, equal weighting is used.

    Raises:
        Exception: If fewer than two loss classes are provided.

    Forward Input:
        preds (Tuple[Tensor]): Tuple of predictions for each task.
        targets (Tuple[Tensor]): Tuple of targets for each task.

    Forward Output:
        Tensor: Scalar weighted sum of all individual losses.
    """

    def __init__(self, losses: list, losses_weights: list = None) -> None:
        super().__init__()

        assert len(losses) > 0

        if len(losses) == 1:
            raise Exception(
                f"Expected number of losses `>=2`, got {len(losses)}."
                f" `CombinationLoss` is used for multi-task training, and requires at least two loss modules."
                f" Use `{losses[0]}` instead."
            )

        if losses_weights is not None:
            assert len(losses) == len(losses_weights)
            self.losses_weights = losses_weights
        else:
            self.losses_weights = [1.0] * len(losses)

        self.losses = nn.ModuleList([Loss() for Loss in losses])

    def forward(self, preds: Tuple[torch.Tensor], targets: Tuple[torch.Tensor]):
        sum_loss = 0.0
        for i, (pred, target, lossfn, weight) in enumerate(
            zip(preds, targets, self.losses, self.losses_weights)
        ):
            sum_loss += lossfn(pred, target) * weight

        return sum_loss


class MousaviLoss(nn.Module):
    """
    Custom loss used in MagNet (Mousavi et al. 2019) and dist-PT Network (Mousavi et al. 2020).

    Forward Input:
        preds (Tensor): Shape (N, 2), where preds[:, 0] are predicted values (y_hat),
                        preds[:, 1] are log-variance estimates (s).
        targets (Tensor): Ground truth values, shape (N,).

    Forward Output:
        Tensor: Scalar loss value.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        y_hat = preds[:, 0].reshape(-1, 1)
        s = preds[:, 1].reshape(-1, 1)
        loss = torch.sum(
            0.5 * torch.exp(-1 * s) * torch.square(torch.abs(targets - y_hat)) + 0.5 * s
        )
        return loss


class NLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for probabilistic models.

    This loss is typically used when the model outputs log-probabilities of a distribution
    (e.g., from log-softmax or mixture models).

    Forward Input:
        preds (Tensor): Log-probabilities of shape (N, C) or (N,).
        targets (Tensor): Ground truth indices (if classification) or values (if density regression).

    Forward Output:
        Tensor: Scalar loss value.
    """

    _epsilon = 1e-6

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        If preds are log-probabilities of targets directly (e.g., log p(y)),
        then simply return the negative log-probabilities.
        """
        loss = -preds  # preds should already be log-likelihoods of correct targets
        if loss.ndim > 1:
            loss = loss.sum(dim=-1)
        return loss.mean()
