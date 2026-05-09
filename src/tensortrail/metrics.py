"""Simple NumPy-based evaluation metrics for TensorTrail."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _to_array(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x)


def accuracy_score(y_true, y_pred) -> float:
    """Multi-class classification accuracy.

    Parameters
    ----------
    y_true:
        Integer class labels, shape ``(n,)``.
    y_pred:
        Predicted class indices ``(n,)`` *or* raw logits / probabilities
        ``(n, C)``.  When 2-D, ``argmax`` is applied along the last axis.

    Returns
    -------
    float
        Fraction of samples where the predicted class matches ``y_true``.

    Example::

        from tensortrail.metrics import accuracy_score
        accuracy_score([0, 1, 2], [0, 2, 2])  # 0.6667
    """
    y_true_arr = _to_array(y_true).reshape(-1).astype(int)
    y_pred_arr = _to_array(y_pred)
    if y_pred_arr.ndim > 1:
        y_pred_arr = np.argmax(y_pred_arr, axis=-1)
    return float(np.mean(y_pred_arr.astype(int) == y_true_arr))


def binary_accuracy(y_true, y_pred, threshold: float = 0.5) -> float:
    """Binary classification accuracy for sigmoid outputs.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 or 1), shape ``(n,)`` or ``(n, 1)``.
    y_pred:
        Predicted probabilities in ``[0, 1]``, shape ``(n,)`` or ``(n, 1)``.
    threshold:
        Decision boundary; predictions ``>= threshold`` are classified as 1.

    Returns
    -------
    float
        Fraction of correctly classified samples.

    Example::

        from tensortrail.metrics import binary_accuracy
        binary_accuracy([0, 1, 1], [0.3, 0.8, 0.6])  # 1.0
    """
    y_true_arr = _to_array(y_true).reshape(-1).astype(int)
    y_pred_arr = _to_array(y_pred).reshape(-1)
    predicted = (y_pred_arr >= threshold).astype(int)
    return float(np.mean(predicted == y_true_arr))
