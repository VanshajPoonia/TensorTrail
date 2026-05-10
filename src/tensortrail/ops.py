"""Reusable tensor operations for TensorTrail."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax."""
    max_values = np.max(x.data, axis=axis, keepdims=True)
    shifted = x - Tensor(max_values)
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax with autograd support."""
    max_values = np.max(x.data, axis=axis, keepdims=True)
    shifted = x - Tensor(max_values)
    log_sum_exp = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - log_sum_exp


def one_hot(labels: np.ndarray | list[int], num_classes: int) -> Tensor:
    """Convert integer labels into a one-hot Tensor."""
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    encoded = np.zeros((labels_array.size, num_classes), dtype=float)
    encoded[np.arange(labels_array.size), labels_array] = 1.0
    return Tensor(encoded)


def accuracy(logits: Tensor | np.ndarray, labels: Tensor | np.ndarray | list[int]) -> float:
    """Compute classification accuracy from logits and integer or one-hot labels."""
    logits_array = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    labels_array = labels.data if isinstance(labels, Tensor) else np.asarray(labels)

    predictions = np.argmax(logits_array, axis=-1)
    if labels_array.ndim > 1:
        targets = np.argmax(labels_array, axis=-1)
    else:
        targets = labels_array.astype(int)
    return float(np.mean(predictions == targets))
