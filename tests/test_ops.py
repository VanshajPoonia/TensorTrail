import numpy as np

from tensortrail import Tensor
from tensortrail.ops import accuracy, log_softmax, one_hot, softmax


def test_basic_operation_forward_values():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([10.0, 20.0])

    np.testing.assert_allclose((x + y).data, [[11, 22], [13, 24]])
    np.testing.assert_allclose((x - 1).data, [[0, 1], [2, 3]])
    np.testing.assert_allclose((x * 2).data, [[2, 4], [6, 8]])
    np.testing.assert_allclose((x / 2).data, [[0.5, 1], [1.5, 2]])
    np.testing.assert_allclose((-x).data, [[-1, -2], [-3, -4]])
    np.testing.assert_allclose((x**2).data, [[1, 4], [9, 16]])


def test_reductions_reshape_transpose_and_activations():
    x = Tensor([[-1.0, 0.0], [1.0, 2.0]])

    assert x.sum().item() == 2.0
    np.testing.assert_allclose(x.mean(axis=0).data, [0.0, 1.0])
    np.testing.assert_allclose(x.reshape(4).data, [-1, 0, 1, 2])
    np.testing.assert_allclose(x.T.data, [[-1, 1], [0, 2]])
    np.testing.assert_allclose(x.relu().data, [[0, 0], [1, 2]])
    np.testing.assert_allclose(x.sigmoid().data, 1 / (1 + np.exp(-x.data)))
    np.testing.assert_allclose(x.tanh().data, np.tanh(x.data))


def test_softmax_log_softmax_one_hot_accuracy():
    logits = Tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    probs = softmax(logits, axis=1)

    np.testing.assert_allclose(probs.data.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(np.exp(log_softmax(logits, axis=1).data), probs.data)
    np.testing.assert_allclose(one_hot([2, 0], 3).data, [[0, 0, 1], [1, 0, 0]])
    assert accuracy(logits, [2, 0]) == 1.0

