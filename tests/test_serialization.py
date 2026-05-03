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
