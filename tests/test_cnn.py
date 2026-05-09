import numpy as np

from tensortrail import (
    AveragePool2D,
    Conv2D,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
    Tensor,
)


def test_conv2d_output_and_parameter_shapes():
    layer = Conv2D(2, 3, kernel_size=3, stride=1, padding=1, seed=0)
    x = Tensor(np.ones((4, 2, 5, 6)))

    out = layer(x)

    assert out.shape == (4, 3, 5, 6)
    assert layer.weight.shape == (3, 2, 3, 3)
    assert layer.bias is not None
    assert layer.bias.shape == (3,)
    assert len(layer.parameters()) == 2


def test_conv2d_backward_produces_gradients():
    layer = Conv2D(1, 2, kernel_size=3, padding=1, seed=1)
    x = Tensor(np.random.default_rng(0).normal(size=(2, 1, 4, 4)), requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_conv2d_matches_finite_difference_for_input_and_weight():
    x = Tensor(np.arange(9.0).reshape(1, 1, 3, 3) / 10, requires_grad=True)
    layer = Conv2D(1, 1, kernel_size=2, bias=False, seed=0)
    layer.weight.data[...] = np.array([[[[0.2, -0.3], [0.5, 0.7]]]])

    loss = layer(x).sum()
    loss.backward()

    eps = 1e-6
    x_index = (0, 0, 1, 1)
    original_x = x.data[x_index]
    x.data[x_index] = original_x + eps
    plus = layer(x).sum().item()
    x.data[x_index] = original_x - eps
    minus = layer(x).sum().item()
    x.data[x_index] = original_x
    numerical_x = (plus - minus) / (2 * eps)

    weight_index = (0, 0, 0, 1)
    original_weight = layer.weight.data[weight_index]
    layer.weight.data[weight_index] = original_weight + eps
    plus = layer(x).sum().item()
    layer.weight.data[weight_index] = original_weight - eps
    minus = layer(x).sum().item()
    layer.weight.data[weight_index] = original_weight
    numerical_weight = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(x.grad[x_index], numerical_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        layer.weight.grad[weight_index],
        numerical_weight,
        rtol=1e-5,
        atol=1e-5,
    )


def test_maxpool2d_output_shape_and_backward_routes_to_maxima():
    layer = MaxPool2D(kernel_size=2)
    x = Tensor(
        np.array(
            [[[[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0],
               [13.0, 14.0, 15.0, 16.0]]]]
        ),
        requires_grad=True,
    )

    out = layer(x)
    out.sum().backward()

    np.testing.assert_allclose(out.data, [[[[6.0, 8.0], [14.0, 16.0]]]])
    assert out.shape == (1, 1, 2, 2)
    expected_grad = np.array(
        [[[[0.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 1.0],
           [0.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 1.0]]]]
    )
    np.testing.assert_allclose(x.grad, expected_grad)


def test_averagepool2d_output_shape_and_backward_distributes_gradients():
    layer = AveragePool2D(kernel_size=2)
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4), requires_grad=True)

    out = layer(x)
    out.sum().backward()

    np.testing.assert_allclose(out.data, [[[[2.5, 4.5], [10.5, 12.5]]]])
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(x.grad, np.full((1, 1, 4, 4), 0.25))


def test_cnn_layers_work_in_sequential_with_flatten_and_linear():
    model = Sequential(
        Conv2D(1, 2, kernel_size=3, padding=1, seed=1),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(18, 3, seed=2),
    )
    x = Tensor(np.random.default_rng(3).normal(size=(2, 1, 6, 6)), requires_grad=True)
    y = Tensor(np.array([0, 2]))

    logits = model(x)
    loss = CrossEntropyLoss()(logits, y)
    loss.backward()

    assert logits.shape == (2, 3)
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None
