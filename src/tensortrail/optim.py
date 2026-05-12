"""Optimizers for TensorTrail parameters."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    """Base optimizer interface."""

    def __init__(self, params: list[Tensor]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        """Update parameters in place."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError("SGD learning rate lr must be non-negative.")
        if momentum < 0:
            raise ValueError("SGD momentum must be non-negative.")
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(param.data) for param in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if self.momentum:
                self._velocity[i] = self.momentum * self._velocity[i] + param.grad
                update = self._velocity[i]
            else:
                update = param.grad
            param.data -= self.lr * update


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError("Adam learning rate lr must be non-negative.")
        if not 0 <= beta1 < 1:
            raise ValueError("Adam beta1 must satisfy 0 <= beta1 < 1.")
        if not 0 <= beta2 < 1:
            raise ValueError("Adam beta2 must satisfy 0 <= beta2 < 1.")
        if eps <= 0:
            raise ValueError("Adam eps must be positive.")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = [np.zeros_like(param.data) for param in self.params]
        self._v = [np.zeros_like(param.data) for param in self.params]

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[i] / (1 - self.beta1**self.t)
            v_hat = self._v[i] / (1 - self.beta2**self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
