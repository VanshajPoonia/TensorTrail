"""Simple NumPy-based model parameter serialization."""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np


def _parameters(model: Any):
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose a parameters() method.")
    return list(model.parameters())


def _named_parameters(model: Any):
    if hasattr(model, "named_parameters"):
        return list(model.named_parameters())
    return [(f"param_{index}", param) for index, param in enumerate(_parameters(model))]


def _buffers(model: Any):
    if not hasattr(model, "buffers"):
        return []
    return list(model.buffers())


def _named_buffers(model: Any):
    if hasattr(model, "named_buffers"):
        return list(model.named_buffers())
    return [(f"buffer_{index}", buffer) for index, buffer in enumerate(_buffers(model))]


def save_model(model: Any, path: str | PathLike[str], named: bool = False) -> None:
    """Save model parameters and simple buffers to a NumPy ``.npz`` file.

    By default, TensorTrail preserves its original positional checkpoint
    format. Passing ``named=True`` stores stable module paths such as
    ``param:layers.0.weight`` for clearer checkpoints.
    """
    if named:
        arrays = {f"param:{name}": param.data for name, param in _named_parameters(model)}
        arrays.update({f"buffer:{name}": buffer for name, buffer in _named_buffers(model)})
    else:
        params = _parameters(model)
        buffers = _buffers(model)
        arrays = {f"param_{index}": param.data for index, param in enumerate(params)}
        arrays.update({f"buffer_{index}": buffer for index, buffer in enumerate(buffers)})
    np.savez(path, **arrays)


def load_model(model: Any, path: str | PathLike[str], strict: bool = True) -> None:
    """Load model parameters and buffers from a NumPy ``.npz`` file."""
    params = _parameters(model)
    buffers = _buffers(model)
    named_params = dict(_named_parameters(model))
    named_buffers = dict(_named_buffers(model))
    with np.load(path) as archive:
        named_param_keys = [key for key in archive.files if key.startswith("param:")]
        named_buffer_keys = [key for key in archive.files if key.startswith("buffer:")]
        if named_param_keys or named_buffer_keys:
            _load_named_arrays(
                archive,
                named_param_keys,
                named_params,
                "param:",
                "parameter",
                strict,
            )
            _load_named_arrays(
                archive,
                named_buffer_keys,
                named_buffers,
                "buffer:",
                "buffer",
                strict,
            )
            return

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


def _load_named_arrays(
    archive,
    archive_keys: list[str],
    model_arrays: dict[str, Any],
    prefix: str,
    label: str,
    strict: bool,
) -> None:
    checkpoint_names = {key.removeprefix(prefix) for key in archive_keys}
    model_names = set(model_arrays)
    if strict and checkpoint_names != model_names:
        missing = sorted(model_names - checkpoint_names)
        unexpected = sorted(checkpoint_names - model_names)
        details = []
        if missing:
            details.append(f"missing {label}s: {missing}")
        if unexpected:
            details.append(f"unexpected {label}s: {unexpected}")
        raise ValueError("checkpoint named state mismatch: " + "; ".join(details))

    for key in sorted(archive_keys):
        name = key.removeprefix(prefix)
        if name not in model_arrays:
            continue
        target = model_arrays[name]
        value = archive[key]
        target_data = target if isinstance(target, np.ndarray) else target.data
        if strict and value.shape != target_data.shape:
            raise ValueError(
                f"{label} {name!r} shape mismatch: checkpoint has {value.shape}, "
                f"model expects {target_data.shape}."
            )
        if value.shape == target_data.shape:
            target_data[...] = value
