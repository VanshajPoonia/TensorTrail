"""Simple NumPy-based model parameter serialization."""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np


def _parameters(model: Any):
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose a parameters() method.")
    return list(model.parameters())


def save_model(model: Any, path: str | PathLike[str]) -> None:
    """Save model parameters to a NumPy ``.npz`` file."""
    params = _parameters(model)
    arrays = {f"param_{index}": param.data for index, param in enumerate(params)}
    np.savez(path, **arrays)


def load_model(model: Any, path: str | PathLike[str], strict: bool = True) -> None:
    """Load model parameters from a NumPy ``.npz`` file into an existing model."""
    params = _parameters(model)
    with np.load(path) as archive:
        keys = sorted(
            (key for key in archive.files if key.startswith("param_")),
            key=lambda key: int(key.split("_", 1)[1]),
        )
        if strict and len(keys) != len(params):
            raise ValueError(
                f"checkpoint has {len(keys)} parameters, but model has {len(params)}."
            )

        for index, (key, param) in enumerate(zip(keys, params)):
            value = archive[key]
            if strict and value.shape != param.data.shape:
                raise ValueError(
                    f"parameter {index} shape mismatch: checkpoint has {value.shape}, "
                    f"model expects {param.data.shape}."
                )
            if value.shape == param.data.shape:
                param.data[...] = value

