"""Tests for metrics.py — accuracy_score and binary_accuracy."""

import numpy as np
import pytest

from tensortrail.metrics import accuracy_score, binary_accuracy
from tensortrail.tensor import Tensor


# ------------------------------------------------------------------ #
# accuracy_score                                                       #
# ------------------------------------------------------------------ #

def test_accuracy_score_perfect():
    assert accuracy_score([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_score_zero():
    assert accuracy_score([0, 0, 0], [1, 2, 1]) == 0.0


def test_accuracy_score_partial():
    result = accuracy_score([0, 1, 2, 1], [0, 2, 2, 1])
    assert abs(result - 0.75) < 1e-9


def test_accuracy_score_with_logits():
    logits = np.array([[2.0, 0.1, 0.1], [0.1, 0.1, 2.0], [0.1, 2.0, 0.1]])
    y_true = [0, 2, 1]
    assert accuracy_score(y_true, logits) == 1.0


def test_accuracy_score_with_tensors():
    logits = Tensor(np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1]]))
    y_true = Tensor(np.array([0, 1, 0]))
    assert accuracy_score(y_true, logits) == 1.0


def test_accuracy_score_returns_float():
    result = accuracy_score([1, 0], [1, 0])
    assert isinstance(result, float)


def test_accuracy_score_1d_pred():
    # predicted class indices directly (no argmax needed)
    assert accuracy_score([0, 1, 2], [0, 1, 2]) == 1.0


# ------------------------------------------------------------------ #
# binary_accuracy                                                      #
# ------------------------------------------------------------------ #

def test_binary_accuracy_perfect():
    assert binary_accuracy([0, 1, 1], [0.2, 0.8, 0.9]) == 1.0


def test_binary_accuracy_all_wrong():
    assert binary_accuracy([0, 0], [0.9, 0.8]) == 0.0


def test_binary_accuracy_partial():
    result = binary_accuracy([0, 1, 0, 1], [0.2, 0.8, 0.9, 0.3])
    assert abs(result - 0.5) < 1e-9


def test_binary_accuracy_custom_threshold():
    # predict 1 when prob >= 0.3
    result = binary_accuracy([1, 1, 0], [0.35, 0.25, 0.6], threshold=0.3)
    # 0.35 >= 0.3 → 1 ✓, 0.25 < 0.3 → 0 ✗, 0.6 >= 0.3 → 1 ✗ → 1/3
    assert abs(result - 1 / 3) < 1e-9


def test_binary_accuracy_with_column_vectors():
    y_true = np.array([[0], [1], [1]])
    y_pred = np.array([[0.3], [0.7], [0.8]])
    assert binary_accuracy(y_true, y_pred) == 1.0


def test_binary_accuracy_with_tensors():
    y_true = Tensor(np.array([0.0, 1.0, 1.0]))
    y_pred = Tensor(np.array([0.2, 0.9, 0.6]))
    assert binary_accuracy(y_true, y_pred) == 1.0


def test_binary_accuracy_returns_float():
    result = binary_accuracy([0, 1], [0.3, 0.7])
    assert isinstance(result, float)
