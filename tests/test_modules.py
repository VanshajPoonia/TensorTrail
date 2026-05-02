import numpy as np
import pytest

from tensortrail import Linear, ReLU, Sequential, Tensor
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
