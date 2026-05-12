import numpy as np
import pytest

from tensortrail import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, Tensor


def test_mse_rejects_shape_mismatch():
    loss = MSELoss()

    with pytest.raises(ValueError, match="shape"):
        loss(Tensor([1.0, 2.0]), Tensor([[1.0, 2.0]]))


def test_binary_cross_entropy_rejects_shape_mismatch():
    loss = BinaryCrossEntropyLoss()

    with pytest.raises(ValueError, match="same shape"):
        loss(Tensor([0.2, 0.8]), Tensor([[0.0, 1.0]]))


def test_binary_cross_entropy_rejects_invalid_probabilities():
    loss = BinaryCrossEntropyLoss()

    with pytest.raises(ValueError, match="probabilities"):
        loss(Tensor([1.2]), Tensor([1.0]))


def test_binary_cross_entropy_rejects_invalid_targets():
    loss = BinaryCrossEntropyLoss()

    with pytest.raises(ValueError, match="targets"):
        loss(Tensor([0.8]), Tensor([2.0]))


def test_cross_entropy_rejects_label_count_mismatch():
    loss = CrossEntropyLoss()

    with pytest.raises(ValueError, match="batch size"):
        loss(Tensor([[1.0, 2.0], [2.0, 1.0]]), np.array([0]))


def test_cross_entropy_rejects_out_of_range_labels():
    loss = CrossEntropyLoss()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        loss(Tensor([[1.0, 2.0]]), np.array([2]))


def test_cross_entropy_rejects_one_hot_shape_mismatch():
    loss = CrossEntropyLoss()

    with pytest.raises(ValueError, match="same shape"):
        loss(Tensor([[1.0, 2.0]]), Tensor([[1.0, 0.0], [0.0, 1.0]]))
