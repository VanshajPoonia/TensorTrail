import numpy as np
import pytest

from tensortrail import (
    BatchNorm1D,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    ReLU,
    Sequential,
    Tensor,
)
from tensortrail.losses import CrossEntropyLoss, MSELoss


def test_linear_parameter_shapes_and_forward_shape():
    layer = Linear(3, 2, seed=0)
    x = Tensor(np.ones((5, 3)))

    out = layer(x)

    assert layer.weight.shape == (3, 2)
    assert layer.bias is not None
    assert layer.bias.shape == (2,)
    assert out.shape == (5, 2)
    assert len(layer.parameters()) == 2


def test_sequential_forward_and_parameters():
    model = Sequential(Linear(2, 4, seed=1), ReLU(), Linear(4, 1, seed=2))
    out = model(Tensor(np.ones((3, 2))))

    assert out.shape == (3, 1)
    assert len(model.parameters()) == 4


def test_flatten_preserves_batch_dimension():
    layer = Flatten()
    x = Tensor(np.ones((4, 2, 3)))

    out = layer(x)

    assert out.shape == (4, 6)


def test_dropout_respects_train_and_eval_modes():
    layer = Dropout(p=0.5, seed=123)
    x = Tensor(np.ones((200,)))

    train_out = layer(x)
    layer.eval()
    eval_out = layer(x)

    assert np.any(train_out.data == 0.0)
    assert train_out.data.mean() == pytest.approx(1.0, abs=0.25)
    np.testing.assert_allclose(eval_out.data, x.data)


def test_batchnorm1d_normalizes_and_tracks_running_stats():
    layer = BatchNorm1D(3, momentum=0.5)
    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ]
        ),
        requires_grad=True,
    )

    out = layer(x)
    out.sum().backward()

    np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(3), atol=1e-7)
    assert not np.allclose(layer.running_mean, np.zeros(3))
    assert len(layer.parameters()) == 2
    assert len(layer.buffers()) == 2
    assert x.grad.shape == x.shape
    assert layer.gamma.grad.shape == (3,)
    assert layer.beta.grad.shape == (3,)

    layer.eval()
    eval_out = layer(Tensor(np.ones((2, 3))))
    assert eval_out.shape == (2, 3)


def test_loss_functions():
    mse = MSELoss()
    loss = mse(Tensor([1.0, 2.0, 3.0]), Tensor([1.0, 1.0, 1.0]))
    assert loss.item() == pytest.approx(5 / 3)

    ce = CrossEntropyLoss()
    logits = Tensor([[3.0, 1.0], [0.2, 2.2]], requires_grad=True)
    targets = np.array([0, 1])
    ce_loss = ce(logits, targets)
    ce_loss.backward()

    assert ce_loss.item() < 0.2
    assert logits.grad.shape == logits.shape


# ------------------------------------------------------------------ #
# LayerNorm                                                            #
# ------------------------------------------------------------------ #

def test_layernorm_output_shape():
    layer = LayerNorm(8)
    x = Tensor(np.random.randn(4, 8))
    out = layer(x)
    assert out.shape == (4, 8)


def test_layernorm_has_parameters():
    layer = LayerNorm(6)
    params = layer.parameters()
    assert len(params) == 2          # gamma and beta
    shapes = {p.shape for p in params}
    assert (6,) in shapes


def test_layernorm_normalizes_each_sample():
    # Each row should have mean ≈ 0 and std ≈ 1 (gamma=1, beta=0 by default).
    layer = LayerNorm(5)
    x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [10.0, 20.0, 30.0, 40.0, 50.0]]))
    out = layer(x)
    row_means = out.data.mean(axis=-1)
    row_stds = out.data.std(axis=-1)
    np.testing.assert_allclose(row_means, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(row_stds, [1.0, 1.0], atol=1e-5)


def test_layernorm_backward_propagates_gradients():
    layer = LayerNorm(4)
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (4,)
    assert layer.beta.grad.shape == (4,)


def test_layernorm_gamma_beta_applied():
    # Setting gamma=2, beta=3 should shift and scale the normalised output.
    layer = LayerNorm(3)
    layer.gamma.data[:] = 2.0
    layer.beta.data[:] = 3.0
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    out = layer(x)
    # Normalised values scaled by 2 and shifted by 3; mean should be 3, not 0.
    assert abs(out.data.mean() - 3.0) < 1e-5


def test_layernorm_rejects_wrong_last_dim():
    layer = LayerNorm(4)
    with pytest.raises(ValueError, match="last dim"):
        layer(Tensor(np.ones((3, 5))))


def test_layernorm_in_sequential():
    model = Sequential(
        Linear(8, 8, seed=1),
        LayerNorm(8),
        ReLU(),
        Linear(8, 3, seed=2),
    )
    x = Tensor(np.random.randn(5, 8))
    out = model(x)
    assert out.shape == (5, 3)
    # gamma and beta from LayerNorm + weights/biases from two Linear layers
    assert len(model.parameters()) == 6
