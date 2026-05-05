"""Simple NumPy-based model parameter serialization."""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np


def _parameters(model: Any):
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose a parameters() method.")
    return list(model.parameters())


def _buffers(model: Any):
    if not hasattr(model, "buffers"):
        return []
    return list(model.buffers())


def save_model(model: Any, path: str | PathLike[str]) -> None:
    """Save model parameters and simple buffers to a NumPy ``.npz`` file."""
    params = _parameters(model)
    buffers = _buffers(model)
    arrays = {f"param_{index}": param.data for index, param in enumerate(params)}
    arrays.update({f"buffer_{index}": buffer for index, buffer in enumerate(buffers)})
    np.savez(path, **arrays)


def load_model(model: Any, path: str | PathLike[str], strict: bool = True) -> None:
    """Load model parameters and buffers from a NumPy ``.npz`` file."""
    params = _parameters(model)
    buffers = _buffers(model)
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

        buffer_keys = sorted(
            (key for key in archive.files if key.startswith("buffer_")),
            key=lambda key: int(key.split("_", 1)[1]),
        )
        if strict and len(buffer_keys) != len(buffers):
            raise ValueError(
                f"checkpoint has {len(buffer_keys)} buffers, but model has {len(buffers)}."
            )

        for index, (key, buffer) in enumerate(zip(buffer_keys, buffers)):
            value = archive[key]
            if strict and value.shape != buffer.shape:
                raise ValueError(
                    f"buffer {index} shape mismatch: checkpoint has {value.shape}, "
                    f"model expects {buffer.shape}."
                )
            if value.shape == buffer.shape:
                buffer[...] = value
