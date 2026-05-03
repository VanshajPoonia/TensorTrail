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
