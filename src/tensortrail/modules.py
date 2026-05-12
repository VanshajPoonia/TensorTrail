"""Small neural network module system for TensorTrail."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .tensor import Tensor


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    """Normalize an integer or length-2 tuple into a pair."""
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return value
    raise TypeError(f"{name} must be an int or a tuple of two ints.")


def _require_4d(x: Tensor, layer_name: str) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"{layer_name} expects input with shape "
            f"(batch, channels, height, width), got {x.shape}."
        )


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
        return [param for _, param in self.named_parameters()]

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        """Return ``(name, parameter)`` pairs owned by this module."""
        params: list[tuple[str, Tensor]] = []
        for name, value in self.__dict__.items():
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor) and value.requires_grad:
                params.append((qualified_name, value))
            elif isinstance(value, Module):
                params.extend(value.named_parameters(qualified_name))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    item_name = f"{qualified_name}.{index}"
                    if isinstance(item, Module):
                        params.extend(item.named_parameters(item_name))
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append((item_name, item))
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
        return [buffer for _, buffer in self.named_buffers()]

    def named_buffers(self, prefix: str = "") -> list[tuple[str, np.ndarray]]:
        """Return ``(name, buffer)`` pairs owned by this module."""
        buffers: list[tuple[str, np.ndarray]] = []
        for name in self._buffers:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                qualified_name = f"{prefix}.{name}" if prefix else name
                buffers.append((qualified_name, value))
        for name, value in self.__dict__.items():
            qualified_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Module):
                buffers.extend(value.named_buffers(qualified_name))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        buffers.extend(item.named_buffers(f"{qualified_name}.{index}"))
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


class Conv2D(Module):
    """Educational 2D convolution for NCHW image tensors.

    The implementation intentionally uses explicit NumPy loops so the forward
    and backward passes are easy to inspect. It supports stride and zero
    padding, but it is not optimized for large images.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        stride_h, stride_w = _pair(stride, "stride")
        pad_h, pad_w = _pair(padding, "padding")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive.")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride values must be positive.")
        if pad_h < 0 or pad_w < 0:
            raise ValueError("padding values must be non-negative.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (pad_h, pad_w)

        fan_in = in_channels * kernel_h * kernel_w
        fan_out = out_channels * kernel_h * kernel_w
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        rng = np.random.default_rng(seed)
        weight = rng.uniform(
            -limit,
            limit,
            size=(out_channels, in_channels, kernel_h, kernel_w),
        )
        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        _require_4d(x, "Conv2D")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Conv2D expected {self.in_channels} input channels, got {channels}."
            )

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        padded_h = height + 2 * pad_h
        padded_w = width + 2 * pad_w
        out_h = (padded_h - kernel_h) // stride_h + 1
        out_w = (padded_w - kernel_w) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "Conv2D kernel is larger than the padded input: "
                f"input={x.shape}, kernel={self.kernel_size}, padding={self.padding}."
            )

        x_padded = np.pad(
            x.data,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        out_data = np.zeros((batch, self.out_channels, out_h, out_w), dtype=float)

        for n in range(batch):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    h_start = oh * stride_h
                    for ow in range(out_w):
                        w_start = ow * stride_w
                        window = x_padded[
                            n,
                            :,
                            h_start : h_start + kernel_h,
                            w_start : w_start + kernel_w,
                        ]
                        out_data[n, oc, oh, ow] = np.sum(window * self.weight.data[oc])
                if self.bias is not None:
                    out_data[n, oc] += self.bias.data[oc]

        children = [x, self.weight]
        if self.bias is not None:
            children.append(self.bias)
        requires_grad = any(child.requires_grad for child in children)
        out = Tensor(out_data, requires_grad=requires_grad, _children=children, _op="conv2d")

        def _backward() -> None:
            if out.grad is None:
                return

            dx_padded = np.zeros_like(x_padded)
            dw = np.zeros_like(self.weight.data)
            db = np.zeros(self.out_channels, dtype=float) if self.bias is not None else None

            for n in range(batch):
                for oc in range(self.out_channels):
                    for oh in range(out_h):
                        h_start = oh * stride_h
                        for ow in range(out_w):
                            w_start = ow * stride_w
                            grad_value = out.grad[n, oc, oh, ow]
                            window = x_padded[
                                n,
                                :,
                                h_start : h_start + kernel_h,
                                w_start : w_start + kernel_w,
                            ]
                            dw[oc] += grad_value * window
                            dx_padded[
                                n,
                                :,
                                h_start : h_start + kernel_h,
                                w_start : w_start + kernel_w,
                            ] += grad_value * self.weight.data[oc]
                            if db is not None:
                                db[oc] += grad_value

            if x.requires_grad:
                if pad_h == 0 and pad_w == 0:
                    dx = dx_padded
                else:
                    dx = dx_padded[:, :, pad_h : pad_h + height, pad_w : pad_w + width]
                x._add_grad(dx)
            self.weight._add_grad(dw)
            if self.bias is not None and db is not None:
                self.bias._add_grad(db)

        out._backward = _backward
        return out


class MaxPool2D(Module):
    """Max pooling over NCHW image tensors."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = self.kernel_size if stride is None else _pair(stride, "stride")
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
            raise ValueError("kernel_size values must be positive.")
        if self.stride[0] <= 0 or self.stride[1] <= 0:
            raise ValueError("stride values must be positive.")

    def forward(self, x: Tensor) -> Tensor:
        _require_4d(x, "MaxPool2D")
        batch, channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (height - kernel_h) // stride_h + 1
        out_w = (width - kernel_w) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "MaxPool2D kernel is larger than the input: "
                f"input={x.shape}, kernel={self.kernel_size}."
            )

        out_data = np.zeros((batch, channels, out_h, out_w), dtype=float)
        max_indices: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        for n in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    h_start = oh * stride_h
                    for ow in range(out_w):
                        w_start = ow * stride_w
                        window = x.data[
                            n,
                            c,
                            h_start : h_start + kernel_h,
                            w_start : w_start + kernel_w,
                        ]
                        flat_index = int(np.argmax(window))
                        local_h, local_w = np.unravel_index(flat_index, window.shape)
                        out_data[n, c, oh, ow] = window[local_h, local_w]
                        max_indices[(n, c, oh, ow)] = (h_start + local_h, w_start + local_w)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,) if x.requires_grad else (),
            _op="maxpool2d",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            dx = np.zeros_like(x.data)
            for (n, c, oh, ow), (h_index, w_index) in max_indices.items():
                dx[n, c, h_index, w_index] += out.grad[n, c, oh, ow]
            x._add_grad(dx)

        out._backward = _backward
        return out


class AveragePool2D(Module):
    """Average pooling over NCHW image tensors."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = self.kernel_size if stride is None else _pair(stride, "stride")
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
            raise ValueError("kernel_size values must be positive.")
        if self.stride[0] <= 0 or self.stride[1] <= 0:
            raise ValueError("stride values must be positive.")

    def forward(self, x: Tensor) -> Tensor:
        _require_4d(x, "AveragePool2D")
        batch, channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (height - kernel_h) // stride_h + 1
        out_w = (width - kernel_w) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "AveragePool2D kernel is larger than the input: "
                f"input={x.shape}, kernel={self.kernel_size}."
            )

        out_data = np.zeros((batch, channels, out_h, out_w), dtype=float)
        for n in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    h_start = oh * stride_h
                    for ow in range(out_w):
                        w_start = ow * stride_w
                        window = x.data[
                            n,
                            c,
                            h_start : h_start + kernel_h,
                            w_start : w_start + kernel_w,
                        ]
                        out_data[n, c, oh, ow] = np.mean(window)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,) if x.requires_grad else (),
            _op="avgpool2d",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            dx = np.zeros_like(x.data)
            scale = 1.0 / (kernel_h * kernel_w)
            for n in range(batch):
                for c in range(channels):
                    for oh in range(out_h):
                        h_start = oh * stride_h
                        for ow in range(out_w):
                            w_start = ow * stride_w
                            dx[
                                n,
                                c,
                                h_start : h_start + kernel_h,
                                w_start : w_start + kernel_w,
                            ] += out.grad[n, c, oh, ow] * scale
            x._add_grad(dx)

        out._backward = _backward
        return out


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
