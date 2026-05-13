import numpy as np
import pytest

from tensortrail import BCEWithLogitsLoss, BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, Tensor


def _manual_bce_with_logits(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


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


def test_bce_with_logits_mean_returns_scalar_tensor():
    loss = BCEWithLogitsLoss()
    logits = Tensor([[0.0], [1.0], [-1.0]], requires_grad=True)
    targets = Tensor([[0.0], [1.0], [0.0]])

    result = loss(logits, targets)

    assert isinstance(result, Tensor)
    assert result.shape == ()


def test_bce_with_logits_reductions():
    logits = Tensor([[0.0], [1.0], [-1.0]], requires_grad=True)
    targets = Tensor([[0.0], [1.0], [0.0]])

    summed = BCEWithLogitsLoss(reduction="sum")(logits, targets)
    elementwise = BCEWithLogitsLoss(reduction="none")(logits, targets)

    assert summed.shape == ()
    assert elementwise.shape == logits.shape
    np.testing.assert_allclose(
        elementwise.data,
        _manual_bce_with_logits(logits.data, targets.data),
    )
    np.testing.assert_allclose(summed.data, elementwise.data.sum())


def test_bce_with_logits_matches_stable_numpy_formula():
    logits = Tensor(np.array([[-2.0], [0.0], [3.0]]), requires_grad=True)
    targets = Tensor(np.array([[0.0], [1.0], [1.0]]))
    expected = _manual_bce_with_logits(logits.data, targets.data).mean()

    result = BCEWithLogitsLoss()(logits, targets)

    np.testing.assert_allclose(result.data, expected)


def test_bce_with_logits_lower_for_confident_correct_predictions():
    targets = Tensor([[0.0], [1.0]])
    correct_logits = Tensor([[-8.0], [8.0]])
    wrong_logits = Tensor([[8.0], [-8.0]])
    loss = BCEWithLogitsLoss()

    assert loss(correct_logits, targets).item() < loss(wrong_logits, targets).item()


def test_bce_with_logits_backward_populates_logits_grad():
    logits = Tensor([[0.0], [2.0], [-2.0]], requires_grad=True)
    targets = Tensor([[0.0], [1.0], [0.0]])
    loss = BCEWithLogitsLoss()

    result = loss(logits, targets)
    result.backward()

    sigmoid = 1 / (1 + np.exp(-logits.data))
    expected = (sigmoid - targets.data) / logits.data.size
    assert logits.grad.shape == logits.shape
    np.testing.assert_allclose(logits.grad, expected)


def test_bce_with_logits_is_stable_for_large_logits():
    logits = Tensor([[1000.0], [-1000.0]], requires_grad=True)
    targets = Tensor([[1.0], [0.0]])
    loss = BCEWithLogitsLoss()

    result = loss(logits, targets)
    result.backward()

    assert np.isfinite(result.data)
    assert np.all(np.isfinite(logits.grad))


def test_bce_with_logits_rejects_invalid_inputs():
    loss = BCEWithLogitsLoss()

    with pytest.raises(ValueError, match="same shape"):
        loss(Tensor([0.0, 1.0]), Tensor([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="targets"):
        loss(Tensor([0.0]), Tensor([2.0]))
    with pytest.raises(ValueError, match="reduction"):
        BCEWithLogitsLoss(reduction="median")


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
