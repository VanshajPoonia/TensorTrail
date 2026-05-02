"""Small neural network module system for TensorTrail."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .tensor import Tensor


class Module:
    """Base class for TensorTrail neural network modules."""

    def __init__(self) -> None:
        self.training = True

    def forward(self, *args, **kwargs):
        """Compute module output. Subclasses must override this method."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        """Return trainable tensor parameters owned by this module."""
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        """Clear gradients on all module parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self) -> None:
        """Set this module and child modules to training mode."""
        self.training = True
        for child in self.children():
            child.train()

    def eval(self) -> None:
        """Set this module and child modules to evaluation mode."""
        self.training = False
        for child in self.children():
            child.eval()

    def children(self) -> list["Module"]:
        """Return immediate child modules."""
        children: list[Module] = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                children.extend(item for item in value if isinstance(item, Module))
        return children


class Linear(Module):
    """Fully connected layer using input @ weight + bias.

    Weight shape is ``(in_features, out_features)`` so a batch with shape
    ``(batch, in_features)`` produces ``(batch, out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """ReLU activation module."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation module."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Tanh activation module."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sequential(Module):
    """Compose modules in order."""

    def __init__(self, *layers: Module | Iterable[Module]) -> None:
        super().__init__()
        if len(layers) == 1 and not isinstance(layers[0], Module):
            self.layers = list(layers[0])  # type: ignore[arg-type]
        else:
            self.layers = list(layers)  # type: ignore[list-item]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

