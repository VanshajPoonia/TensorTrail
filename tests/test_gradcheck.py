import numpy as np
import pytest

from tensortrail import Tensor, gradcheck


def test_gradcheck_passes_for_smooth_expression():
    x = Tensor(np.array([[0.2, -0.4], [0.7, 1.1]]), requires_grad=True)
    y = Tensor(np.array([[1.0, -0.3], [0.5, 0.8]]), requires_grad=True)

    result = gradcheck(lambda a, b: ((a * b).tanh() + a.exp()).mean(), [x, y])

    assert result.passed
    assert result.max_abs_error < 1e-5
    assert result.failures == []


def test_gradcheck_passes_for_scalar_square_sum():
    x = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)

    result = gradcheck(lambda value: (value * value).sum(), x)

    assert result.passed
    assert result.max_abs_error < 1e-5


def test_gradcheck_reports_failure_for_nonsmooth_point():
    x = Tensor([0.0], requires_grad=True)

    result = gradcheck(lambda value: value.relu().sum(), x)

    assert not result.passed
    assert result.max_abs_error > 0.1
    assert result.failures


def test_gradcheck_can_raise_on_failure():
    x = Tensor([0.0], requires_grad=True)

    with pytest.raises(AssertionError):
        gradcheck(lambda value: value.relu().sum(), x, raise_on_fail=True)
