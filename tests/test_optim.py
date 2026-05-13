import numpy as np
import pytest

from tensortrail import Adam, SGD, Tensor


def test_sgd_updates_parameters():
    p = Tensor([1.0, 2.0], requires_grad=True)
    loss = (p * p).sum()
    loss.backward()

    opt = SGD([p], lr=0.1)
    opt.step()

    np.testing.assert_allclose(p.data, [0.8, 1.6])


def test_sgd_momentum_updates_parameters():
    p = Tensor([1.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)

    p.grad = np.array([2.0])
    opt.step()
    p.grad = np.array([2.0])
    opt.step()

    np.testing.assert_allclose(p.data, [0.42])


def test_adam_updates_parameters():
    p = Tensor([1.0, -1.0], requires_grad=True)
    p.grad = np.array([0.5, -0.5])

    opt = Adam([p], lr=0.01)
    opt.step()

    assert p.data[0] < 1.0
    assert p.data[1] > -1.0

    opt.zero_grad()
    assert p.grad is None


def test_clip_grad_norm_limits_global_norm():
    p1 = Tensor([0.0, 0.0], requires_grad=True)
    p2 = Tensor([0.0], requires_grad=True)
    p1.grad = np.array([3.0, 4.0])
    p2.grad = np.array([12.0])
    opt = SGD([p1, p2], lr=0.1)

    original_norm = opt.clip_grad_norm(1.0)
    clipped_norm = np.sqrt(np.sum(p1.grad * p1.grad) + np.sum(p2.grad * p2.grad))

    assert original_norm == pytest.approx(13.0)
    assert clipped_norm == pytest.approx(1.0)


def test_clip_grad_norm_affects_next_step():
    p = Tensor([1.0], requires_grad=True)
    p.grad = np.array([10.0])
    opt = SGD([p], lr=0.1)

    opt.clip_grad_norm(1.0)
    opt.step()

    np.testing.assert_allclose(p.data, [0.9])


def test_clip_grad_norm_rejects_invalid_max_norm():
    p = Tensor([1.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    with pytest.raises(ValueError, match="max_norm"):
        opt.clip_grad_norm(0.0)


def test_sgd_rejects_invalid_hyperparameters():
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="learning rate"):
        SGD([p], lr=-0.1)
    with pytest.raises(ValueError, match="momentum"):
        SGD([p], momentum=-0.1)


def test_adam_rejects_invalid_hyperparameters():
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="learning rate"):
        Adam([p], lr=-0.1)
    with pytest.raises(ValueError, match="beta1"):
        Adam([p], beta1=1.0)
    with pytest.raises(ValueError, match="beta2"):
        Adam([p], beta2=-0.1)
    with pytest.raises(ValueError, match="eps"):
        Adam([p], eps=0.0)


def test_xor_training_reduces_loss():
    from tensortrail import BinaryCrossEntropyLoss, Linear, Sequential, Sigmoid, Tanh
    from tensortrail.data import make_xor

    dataset = make_xor(repeats=16, noise=0.01, seed=4)
    model = Sequential(Linear(2, 6, seed=1), Tanh(), Linear(6, 1, seed=2), Sigmoid())
    loss_fn = BinaryCrossEntropyLoss()
    opt = Adam(model.parameters(), lr=0.05)
    x = Tensor(dataset.x)
    y = Tensor(dataset.y)

    initial = loss_fn(model(x), y).item()
    for _ in range(300):
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = loss_fn(model(x), y).item()

    assert final < initial * 0.5
