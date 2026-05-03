import numpy as np
import pytest

from tensortrail import Linear, ReLU, Sequential, load_model, save_model


def test_save_load_round_trips_sequential_parameters(tmp_path):
    model = Sequential(Linear(2, 3, seed=1), ReLU(), Linear(3, 1, seed=2))
    original = [param.data.copy() for param in model.parameters()]
    path = tmp_path / "model.npz"

    save_model(model, path)
    for param in model.parameters():
        param.data += 10.0

    load_model(model, path)

    for param, expected in zip(model.parameters(), original):
        np.testing.assert_allclose(param.data, expected)


def test_load_model_strict_parameter_count_error(tmp_path):
    source = Sequential(Linear(2, 3, seed=1))
    target = Sequential(Linear(2, 3, seed=1), Linear(3, 1, seed=2))
    path = tmp_path / "count_mismatch.npz"

    save_model(source, path)

    with pytest.raises(ValueError, match="checkpoint has"):
        load_model(target, path)


def test_load_model_strict_shape_error(tmp_path):
    source = Sequential(Linear(2, 3, seed=1))
    target = Sequential(Linear(2, 4, seed=1))
    path = tmp_path / "shape_mismatch.npz"

    save_model(source, path)

    with pytest.raises(ValueError, match="shape mismatch"):
        load_model(target, path)

