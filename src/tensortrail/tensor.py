"""Core Tensor object and reverse-mode autograd engine for TensorTrail."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np


ArrayLike = Any


def _ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input data to a floating NumPy array."""
    if isinstance(data, Tensor):
        return data.data
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"Tensor data must be real numeric values, got dtype {array.dtype}."
        )
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(
            f"Tensor data must be real numeric values; complex dtype {array.dtype} "
            "is not supported."
        )
    return array.astype(float)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to an operand's original shape."""
    grad = np.asarray(grad, dtype=float)

    if shape == ():
        return np.asarray(grad.sum(), dtype=float)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


def _as_tensor(value: ArrayLike) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


def _matmul_data(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Run matmul while avoiding noisy false-positive BLAS warning flags."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def _children_if_tracking(requires_grad: bool, *children: "Tensor") -> tuple["Tensor", ...]:
    return tuple(children) if requires_grad else ()


class Tensor:
    """A NumPy-backed tensor with reverse-mode automatic differentiation.

    TensorTrail records a dynamic computation graph whenever an operation uses
    a tensor whose ``requires_grad`` flag is true. Calling ``backward`` walks
    that graph in reverse topological order and accumulates gradients into
    leaf tensors.
    """

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        *,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = _ensure_array(data)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor's shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of tensor dimensions."""
        return self.data.ndim

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, "
            f"requires_grad={self.requires_grad}, shape={self.shape})"
        )

    def __hash__(self) -> int:
        return id(self)

    def _add_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=float)
        self.grad = grad if self.grad is None else self.grad + grad

    def zero_grad(self) -> None:
        """Clear this tensor's accumulated gradient."""
        self.grad = None

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Backpropagate through the computation graph.

        Scalar tensors can call ``backward()`` directly. Non-scalar tensors
        require an external gradient with the same shape as ``self.data``.
        """
        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    "backward() called on a non-scalar Tensor with shape "
                    f"{self.shape}; pass an external gradient with the same shape."
                )
            grad_array = np.ones_like(self.data)
        else:
            try:
                grad_array = _ensure_array(grad)
            except TypeError as exc:
                raise TypeError("backward() gradient must be real numeric values.") from exc
            if grad_array.shape != self.data.shape:
                raise ValueError(
                    f"backward() gradient shape {grad_array.shape} does not match "
                    f"Tensor shape {self.data.shape}."
                )

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = grad_array

        for node in reversed(topo):
            node._backward()

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = _as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.data + other.data,
            requires_grad=requires_grad,
            _children=_children_if_tracking(requires_grad, self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(out.grad, self.shape))
            other._add_grad(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-_as_tensor(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return _as_tensor(other) - self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = _as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.data * other.data,
            requires_grad=requires_grad,
            _children=_children_if_tracking(requires_grad, self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(out.grad * other.data, self.shape))
            other._add_grad(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = _as_tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return _as_tensor(other) / self

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="neg",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(-out.grad)

        out._backward = _backward
        return out

    def __pow__(self, power: float | int) -> "Tensor":
        if isinstance(power, Tensor):
            raise TypeError("Tensor powers are not supported; use a numeric exponent.")
        out = Tensor(
            self.data**power,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op=f"pow({power})",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = _as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            _matmul_data(self.data, other.data),
            requires_grad=requires_grad,
            _children=_children_if_tracking(requires_grad, self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self_grad = _matmul_data(out.grad, np.swapaxes(other.data, -1, -2))
            other_grad = _matmul_data(np.swapaxes(self.data, -1, -2), out.grad)
            self._add_grad(_unbroadcast(self_grad, self.shape))
            other._add_grad(_unbroadcast(other_grad, other.shape))

        out._backward = _backward
        return out

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return _as_tensor(other) @ self

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        """Sum tensor elements along optional axes."""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self._add_grad(np.broadcast_to(grad, self.shape))

        out._backward = _backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        """Average tensor elements along optional axes."""
        if axis is None:
            divisor = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            divisor = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        """Return a tensor with the same data and a new shape."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            new_shape = shape[0]
        else:
            new_shape = tuple(shape)  # type: ignore[arg-type]
        out = Tensor(
            self.data.reshape(new_shape),
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> "Tensor":
        """Return a tensor with axes transposed."""
        out = Tensor(
            np.transpose(self.data, axes=axes),
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if axes is None:
                inverse_axes = None
            else:
                inverse_axes = np.argsort(axes)
            self._add_grad(np.transpose(out.grad, axes=inverse_axes))

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        """Reverse tensor axes, matching NumPy's ``.T`` behavior."""
        return self.transpose()

    def exp(self) -> "Tensor":
        """Elementwise exponential."""
        data = np.exp(self.data)
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="exp",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Elementwise natural logarithm."""
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="log",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """Elementwise hyperbolic tangent."""
        data = np.tanh(self.data)
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="tanh",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * (1 - data**2))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Elementwise rectified linear unit."""
        data = np.maximum(self.data, 0)
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        """Elementwise logistic sigmoid."""
        data = np.where(
            self.data >= 0,
            1 / (1 + np.exp(-self.data)),
            np.exp(self.data) / (1 + np.exp(self.data)),
        )
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=_children_if_tracking(self.requires_grad, self),
            _op="sigmoid",
        )

        def _backward() -> None:
            if out.grad is not None:
                self._add_grad(out.grad * data * (1 - data))

        out._backward = _backward
        return out

    def item(self) -> float:
        """Return the scalar value as a Python float."""
        return float(self.data.item())


def tensor(data: ArrayLike, requires_grad: bool = False) -> Tensor:
    """Convenience factory for creating TensorTrail tensors."""
    return Tensor(data, requires_grad=requires_grad)
