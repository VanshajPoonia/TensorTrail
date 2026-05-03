"""Finite-difference gradient checking for TensorTrail."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .tensor import Tensor


@dataclass
class GradCheckResult:
    """Result returned by ``gradcheck``."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    failures: list[str] = field(default_factory=list)


def _normalize_inputs(inputs: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if isinstance(inputs, Tensor):
        return [inputs]
    normalized = list(inputs)
    if not normalized or not all(isinstance(item, Tensor) for item in normalized):
        raise TypeError("gradcheck inputs must be a Tensor or a non-empty sequence of Tensors.")
    return normalized


def _scalar_value(output: Tensor) -> float:
    if not isinstance(output, Tensor):
        raise TypeError("gradcheck function must return a Tensor.")
    if output.data.size != 1:
        raise ValueError("gradcheck function must return a scalar Tensor.")
    return output.item()


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Tensor | Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    raise_on_fail: bool = False,
) -> GradCheckResult:
    """Compare TensorTrail analytical gradients with finite differences.

    ``fn`` must accept the provided Tensor inputs and return a scalar Tensor.
    The checker temporarily marks inputs as requiring gradients, computes
    analytical gradients with ``backward()``, then estimates numerical
    gradients with centered finite differences.
    """
    tensors = _normalize_inputs(inputs)
    original_requires_grad = [tensor.requires_grad for tensor in tensors]
    original_grads = [None if tensor.grad is None else tensor.grad.copy() for tensor in tensors]
    original_data = [tensor.data.copy() for tensor in tensors]

    try:
        for tensor in tensors:
            tensor.requires_grad = True
            tensor.zero_grad()

        output = fn(*tensors)
        _scalar_value(output)
        output.backward()

        analytical = [
            np.zeros_like(tensor.data) if tensor.grad is None else tensor.grad.copy()
            for tensor in tensors
        ]

        numerical: list[np.ndarray] = []
        for tensor_index, tensor in enumerate(tensors):
            grad = np.zeros_like(tensor.data)
            for index in np.ndindex(tensor.data.shape):
                tensor.data[index] += eps
                plus = _scalar_value(fn(*tensors))
                tensor.data[index] -= 2 * eps
                minus = _scalar_value(fn(*tensors))
                tensor.data[index] += eps
                grad[index] = (plus - minus) / (2 * eps)
            numerical.append(grad)
            tensor.data[...] = original_data[tensor_index]

        failures: list[str] = []
        max_abs_error = 0.0
        max_rel_error = 0.0

        for tensor_index, (actual, expected) in enumerate(zip(analytical, numerical)):
            abs_error = np.abs(actual - expected)
            rel_error = abs_error / np.maximum(np.abs(expected), atol)
            max_abs_error = max(max_abs_error, float(abs_error.max(initial=0.0)))
            max_rel_error = max(max_rel_error, float(rel_error.max(initial=0.0)))

            mismatch = ~np.isclose(actual, expected, atol=atol, rtol=rtol)
            if np.any(mismatch):
                first = tuple(np.argwhere(mismatch)[0])
                failures.append(
                    "input "
                    f"{tensor_index} index {first}: analytical={actual[first]:.8f}, "
                    f"numerical={expected[first]:.8f}"
                )

        passed = not failures
        result = GradCheckResult(
            passed=passed,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            failures=failures,
        )
        if raise_on_fail and not passed:
            raise AssertionError(
                "gradcheck failed: "
                f"max_abs_error={max_abs_error:.6g}, max_rel_error={max_rel_error:.6g}; "
                + "; ".join(failures[:3])
            )
        return result
    finally:
        for tensor, data, grad, requires_grad in zip(
            tensors, original_data, original_grads, original_requires_grad
        ):
            tensor.data[...] = data
            tensor.grad = grad
            tensor.requires_grad = requires_grad

