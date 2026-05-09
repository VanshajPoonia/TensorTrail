"""Small neural network module system for TensorTrail."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .tensor import Tensor


class Module:
    """Base class for TensorTrail neural network modules."""

    def __init__(self) -> None:
        self.training = True
        self._buffers: list[str] = []

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

    def buffers(self) -> list[np.ndarray]:
        """Return non-trainable state arrays owned by this module.

        Buffers are values such as BatchNorm running statistics: they are not
        optimized by gradient descent, but they are part of a model checkpoint.
        """
        buffers: list[np.ndarray] = []
        for name in self._buffers:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                buffers.append(value)
        for child in self.children():
            buffers.extend(child.buffers())
        return buffers

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


class Flatten(Module):
    """Flatten each sample in a batch into one feature vector."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            raise ValueError("Flatten expects at least one dimension.")
        batch = x.shape[0]
        return x.reshape(batch, -1)


class Dropout(Module):
    """Inverted dropout regularization.

    During training, elements are randomly zeroed and the remaining activations
    are scaled by ``1 / (1 - p)``. During evaluation, dropout is an identity.
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("dropout probability p must satisfy 0 <= p < 1.")
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        keep_probability = 1.0 - self.p
        mask = (self.rng.random(x.shape) < keep_probability).astype(float)
        return x * Tensor(mask / keep_probability)


class BatchNorm1D(Module):
    """Simple batch normalization for 2D ``(batch, features)`` tensors."""

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if not 0 < momentum <= 1:
            raise ValueError("momentum must satisfy 0 < momentum <= 1.")
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self._buffers.extend(["running_mean", "running_var"])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("BatchNorm1D expects input with shape (batch, features).")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm1D expected {self.num_features} features, got {x.shape[1]}."
            )

        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=0, keepdims=True)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data.reshape(-1)
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var.data.reshape(-1)
            )
        else:
            mean = Tensor(self.running_mean.reshape(1, -1))
            var = Tensor(self.running_var.reshape(1, -1))
            centered = x - mean

        normalized = centered / ((var + self.eps) ** 0.5)
        return normalized * self.gamma + self.beta


class LayerNorm(Module):
    """Layer normalization over the last dimension.

    Unlike :class:`BatchNorm1D`, which normalizes across the batch for each
    feature, LayerNorm normalizes each *sample* across its own feature
    dimension.  This makes it independent of batch size and well-suited to
    small batches or sequence models.

    For a 2-D input of shape ``(batch, features)`` the normalization is
    applied independently per row::

        mean  = x.mean(axis=-1, keepdims=True)
        var   = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (x - mean) / sqrt(var + eps)
        out   = gamma * x_hat + beta

    ``gamma`` and ``beta`` are learnable per-feature scale and shift parameters
    that appear in :meth:`Module.parameters`.

    Reference: Ba et al., "Layer Normalization" (2016).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be a positive integer.")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected last dim {self.normalized_shape}, "
                f"got {x.shape[-1]}."
            )
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / ((var + self.eps) ** 0.5)
        return normalized * self.gamma + self.beta


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
