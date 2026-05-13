"""Loss functions for TensorTrail."""

from __future__ import annotations

import numpy as np

from .ops import log_softmax, one_hot
from .tensor import Tensor


def _stable_sigmoid(data: np.ndarray) -> np.ndarray:
    """Compute sigmoid without overflow warnings for very large logits."""
    result = np.empty_like(data, dtype=float)
    positive = data >= 0
    result[positive] = 1 / (1 + np.exp(-data[positive]))
    exp_values = np.exp(data[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


class MSELoss:
    """Mean squared error loss."""

    def __call__(self, predictions: Tensor, targets: Tensor | np.ndarray) -> Tensor:
        target_tensor = targets if isinstance(targets, Tensor) else Tensor(targets)
        if predictions.shape != target_tensor.shape:
            raise ValueError(
                f"MSELoss predictions shape {predictions.shape} must match "
                f"targets shape {target_tensor.shape}."
            )
        diff = predictions - target_tensor
        return (diff * diff).mean()


class BinaryCrossEntropyLoss:
    """Binary cross entropy loss for probabilities."""

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps

    def __call__(self, predictions: Tensor, targets: Tensor | np.ndarray) -> Tensor:
        target_tensor = targets if isinstance(targets, Tensor) else Tensor(targets)
        if predictions.shape != target_tensor.shape:
            raise ValueError(
                "BinaryCrossEntropyLoss predictions and targets must have the same shape."
            )
        if np.any((predictions.data < 0) | (predictions.data > 1)):
            raise ValueError(
                "BinaryCrossEntropyLoss expects prediction probabilities in [0, 1]."
            )
        if np.any((target_tensor.data < 0) | (target_tensor.data > 1)):
            raise ValueError("BinaryCrossEntropyLoss targets must be in [0, 1].")
        clipped = predictions * (1 - 2 * self.eps) + self.eps
        return -(
            target_tensor * clipped.log()
            + (1 - target_tensor) * (1 - clipped).log()
        ).mean()


class BCEWithLogitsLoss:
    """Binary cross entropy loss for raw logits.

    This combines a sigmoid activation and binary cross entropy in one
    numerically stable operation. Use it when the model's final layer returns
    unconstrained logits rather than probabilities.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("BCEWithLogitsLoss reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: Tensor | np.ndarray) -> Tensor:
        target_tensor = targets if isinstance(targets, Tensor) else Tensor(targets)
        if logits.shape != target_tensor.shape:
            raise ValueError(
                "BCEWithLogitsLoss logits and targets must have the same shape."
            )
        if np.any((target_tensor.data < 0) | (target_tensor.data > 1)):
            raise ValueError("BCEWithLogitsLoss targets must be in [0, 1].")

        elementwise = (
            np.maximum(logits.data, 0)
            - logits.data * target_tensor.data
            + np.log1p(np.exp(-np.abs(logits.data)))
        )
        if self.reduction == "mean":
            data = np.asarray(elementwise.mean(), dtype=float)
        elif self.reduction == "sum":
            data = np.asarray(elementwise.sum(), dtype=float)
        else:
            data = elementwise

        requires_grad = logits.requires_grad or target_tensor.requires_grad
        children = (logits, target_tensor) if requires_grad else ()
        out = Tensor(data, requires_grad=requires_grad, _children=children, _op="bce_logits")

        def _backward() -> None:
            if out.grad is None:
                return
            sigmoid = _stable_sigmoid(logits.data)
            grad = out.grad
            if self.reduction == "mean":
                grad = np.broadcast_to(grad / logits.data.size, logits.shape)
            elif self.reduction == "sum":
                grad = np.broadcast_to(grad, logits.shape)
            logits._add_grad(grad * (sigmoid - target_tensor.data))
            target_tensor._add_grad(grad * -logits.data)

        out._backward = _backward
        return out


class CrossEntropyLoss:
    """Multi-class cross entropy loss from raw logits and integer labels."""

    def __call__(self, logits: Tensor, targets: Tensor | np.ndarray | list[int]) -> Tensor:
        if isinstance(targets, Tensor):
            target_data = targets.data
        else:
            target_data = np.asarray(targets)

        if target_data.ndim == 1:
            if logits.ndim < 2:
                raise ValueError("CrossEntropyLoss expects logits with class dimension.")
            labels = target_data.astype(int)
            if not np.allclose(target_data, labels):
                raise ValueError("CrossEntropyLoss class labels must be integers.")
            if np.any((labels < 0) | (labels >= logits.shape[-1])):
                raise ValueError(
                    f"CrossEntropyLoss labels must be in [0, {logits.shape[-1] - 1}]."
                )
            expected_batch = logits.shape[0] if logits.ndim > 1 else 1
            if labels.shape[0] != expected_batch:
                raise ValueError(
                    "CrossEntropyLoss label count must match the logits batch size."
                )
            target_tensor = one_hot(labels, logits.shape[-1])
        else:
            target_tensor = Tensor(target_data)
            if target_tensor.shape != logits.shape:
                raise ValueError(
                    "CrossEntropyLoss one-hot targets must have the same shape as logits."
                )

        log_probs = log_softmax(logits, axis=-1)
        per_example = -(target_tensor * log_probs).sum(axis=-1)
        return per_example.mean()
