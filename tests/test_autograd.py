import numpy as np

from tensortrail import Tensor


def test_basic_operation_gradients_accumulate_from_branches():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x + x).sum()

    y.backward()

    np.testing.assert_allclose(x.grad, [3.0, 5.0, 7.0])


def test_addition_gradient():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([0.5, -1.0, 2.0], requires_grad=True)

    ((x + y) * Tensor([1.0, 2.0, 3.0])).sum().backward()

    np.testing.assert_allclose(x.grad, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y.grad, [1.0, 2.0, 3.0])


def test_multiplication_gradient():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    (x * y).sum().backward()

    np.testing.assert_allclose(x.grad, y.data)
    np.testing.assert_allclose(y.grad, x.data)


def test_division_gradient():
    x = Tensor([2.0, 4.0, 8.0], requires_grad=True)
    y = Tensor([1.0, 2.0, 4.0], requires_grad=True)

    (x / y).sum().backward()

    np.testing.assert_allclose(x.grad, 1 / y.data)
    np.testing.assert_allclose(y.grad, -x.data / (y.data**2))


def test_power_gradient():
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)

    (x**3).sum().backward()

    np.testing.assert_allclose(x.grad, 3 * x.data**2)


def test_broadcasting_gradients():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = Tensor([10.0, 20.0, 30.0], requires_grad=True)

    y = (x + b).sum()
    y.backward()

    np.testing.assert_allclose(x.grad, np.ones_like(x.data))
    np.testing.assert_allclose(b.grad, [2.0, 2.0, 2.0])


def test_matmul_gradients():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[2.0, 0.0], [1.0, -1.0]], requires_grad=True)

    y = (x @ w).sum()
    y.backward()

    np.testing.assert_allclose(x.grad, np.ones((2, 2)) @ w.data.T)
    np.testing.assert_allclose(w.grad, x.data.T @ np.ones((2, 2)))


def test_sum_and_mean_gradients_with_axes():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    x.sum(axis=0).backward(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(x.grad, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    x.zero_grad()
    x.mean(axis=1, keepdims=True).backward(np.array([[2.0], [4.0]]))
    np.testing.assert_allclose(
        x.grad,
        [[2 / 3, 2 / 3, 2 / 3], [4 / 3, 4 / 3, 4 / 3]],
    )


def test_reshape_and_transpose_gradients():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)

    x.reshape(3, 2).backward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    np.testing.assert_allclose(x.grad, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    x.zero_grad()
    x.transpose().backward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    np.testing.assert_allclose(x.grad, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


def test_exp_and_log_gradients():
    x = Tensor([1.0, 2.0, 4.0], requires_grad=True)

    (x.exp() + x.log()).sum().backward()

    np.testing.assert_allclose(x.grad, np.exp(x.data) + 1 / x.data)


def test_activation_gradients():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = (x.relu() + x.sigmoid() + x.tanh()).sum()
    y.backward()

    sigmoid = 1 / (1 + np.exp(-x.data))
    expected = (x.data > 0) + sigmoid * (1 - sigmoid) + (1 - np.tanh(x.data) ** 2)
    np.testing.assert_allclose(x.grad, expected)


def test_finite_difference_gradient_check():
    data = np.array([[0.2, -0.4], [0.7, 1.1]], dtype=float)
    x = Tensor(data.copy(), requires_grad=True)

    y = ((x * x).tanh().sum() + (x.exp().mean())) / 3.0
    y.backward()

    eps = 1e-6
    numerical = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plus = data.copy()
            minus = data.copy()
            plus[i, j] += eps
            minus[i, j] -= eps

            f_plus = (
                np.tanh(plus * plus).sum()
                + np.exp(plus).mean()
            ) / 3.0
            f_minus = (
                np.tanh(minus * minus).sum()
                + np.exp(minus).mean()
            ) / 3.0
            numerical[i, j] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(x.grad, numerical, rtol=1e-5, atol=1e-5)
