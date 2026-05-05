import numpy as np
import pytest

from tensortrail import Tensor


def test_tensor_creation_shape_repr_and_zero_grad():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)

    assert x.shape == (2, 2)
    assert "Tensor" in repr(x)
    assert x.requires_grad

    y = (x * 2).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, np.full((2, 2), 2.0))

    x.zero_grad()
    assert x.grad is None


def test_backward_requires_gradient_for_non_scalar():
    x = Tensor([1, 2, 3], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()


def test_non_scalar_backward_with_external_gradient():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x * x
    y.backward(np.ones(3))
    np.testing.assert_allclose(x.grad, [2, 4, 6])


def test_tensor_rejects_unsupported_dtypes():
    with pytest.raises(TypeError, match="real numeric"):
        Tensor(["not", "numbers"])

    with pytest.raises(TypeError, match="complex dtype"):
        Tensor(np.array([1 + 2j]))
