"""Readable training loop utilities for TensorTrail."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .data import DataLoader, Dataset
from .ops import accuracy
from .tensor import Tensor


MetricFn = Callable[[Tensor, Tensor], float]


class Trainer:
    """Small supervised training loop for TensorTrail models."""

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        metric_fn: MetricFn | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn

    def fit(
        self,
        dataset: Dataset,
        epochs: int = 20,
        batch_size: int = 32,
        shuffle: bool = True,
        print_every: int | None = None,
    ) -> dict[str, list[float]]:
        """Train a model and return history."""
        history: dict[str, list[float]] = {"loss": []}
        if self.metric_fn is not None:
            history["metric"] = []

        for epoch in range(1, epochs + 1):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, seed=epoch)
            losses: list[float] = []
            metrics: list[float] = []

            for x_batch, y_batch in loader:
                predictions = self.model(x_batch)
                loss = self.loss_fn(predictions, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if self.metric_fn is not None:
                    metrics.append(self.metric_fn(predictions, y_batch))

            mean_loss = float(np.mean(losses))
            history["loss"].append(mean_loss)
            message = f"epoch {epoch:03d} loss={mean_loss:.4f}"
            if self.metric_fn is not None:
                mean_metric = float(np.mean(metrics))
                history["metric"].append(mean_metric)
                message += f" metric={mean_metric:.4f}"
            if print_every is not None and (epoch == 1 or epoch % print_every == 0):
                print(message)

        return history


def classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Metric helper for multi-class classification."""
    return accuracy(logits, labels)

