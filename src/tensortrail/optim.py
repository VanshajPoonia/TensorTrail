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

    def clip_grad_norm(self, max_norm: float, eps: float = 1e-12) -> float:
        """Clip gradients by global L2 norm and return the pre-clip norm."""
        if max_norm <= 0:
            raise ValueError("max_norm must be positive.")

        squared_norm = 0.0
        for param in self.params:
            if param.grad is not None:
                squared_norm += float(np.sum(param.grad * param.grad))

        total_norm = float(np.sqrt(squared_norm))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + eps)
            for param in self.params:
                if param.grad is not None:
                    param.grad *= scale
        return total_norm

    def step(self) -> None:
        """Update parameters in place."""
        raise NotImplementedError


class LRScheduler:
    """Base class for simple epoch-level learning-rate schedules."""

    def __init__(self, optimizer: Optimizer) -> None:
        if not hasattr(optimizer, "lr"):
            raise TypeError("Learning-rate schedulers require an optimizer with an lr.")
        self.optimizer = optimizer
        self.last_epoch = 0

    def get_lr(self) -> float:
        """Return the optimizer's current learning rate."""
        return float(self.optimizer.lr)

    def step(self) -> float:
        """Advance the schedule by one epoch and return the new learning rate."""
        raise NotImplementedError


class StepLR(LRScheduler):
    """Decay the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        if step_size <= 0:
            raise ValueError("StepLR step_size must be positive.")
        if gamma <= 0:
            raise ValueError("StepLR gamma must be positive.")
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> float:
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.get_lr()


class ExponentialLR(LRScheduler):
    """Decay the learning rate by ``gamma`` every epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float) -> None:
        if gamma <= 0:
            raise ValueError("ExponentialLR gamma must be positive.")
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> float:
        self.last_epoch += 1
        self.optimizer.lr *= self.gamma
        return self.get_lr()


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
