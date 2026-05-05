"""Readable training loop utilities for TensorTrail."""

from __future__ import annotations

import time
from collections.abc import Mapping
from os import PathLike
from typing import Callable, Optional

import numpy as np

from .data import DataLoader, Dataset
from .ops import accuracy
from .serialization import save_model
from .tensor import Tensor


MetricFn = Callable[[Tensor, Tensor], float]
Metrics = Optional[Mapping[str, MetricFn]]


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
        val_dataset: Dataset | None = None,
        metrics: Metrics = None,
        early_stopping_patience: int | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
    ) -> dict[str, list[float]]:
        """Train a model and return history."""
        metric_fns = self._resolve_metrics(metrics)
        train_losses: list[float] = []
        elapsed_times: list[float] = []
        epochs_seen: list[float] = []
        history: dict[str, list[float]] = {
            "epoch": epochs_seen,
            "train_loss": train_losses,
            "elapsed_time": elapsed_times,
            "loss": train_losses,
        }
        if val_dataset is not None:
            history["val_loss"] = []
        for name in metric_fns:
            history[name] = []
            if val_dataset is not None:
                history[f"val_{name}"] = []
        if metric_fns:
            first_metric = next(iter(metric_fns))
            history["metric"] = history[first_metric]

        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_result = self._run_epoch(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=epoch,
                metric_fns=metric_fns,
                training=True,
            )

            mean_loss = train_result["loss"]
            elapsed = time.perf_counter() - start_time
            history["epoch"].append(float(epoch))
            history["train_loss"].append(mean_loss)
            history["elapsed_time"].append(elapsed)
            message = f"epoch {epoch:03d} train_loss={mean_loss:.4f}"
            for name in metric_fns:
                value = train_result[name]
                history[name].append(value)
                message += f" {name}={value:.4f}"

            monitor_loss = mean_loss
            if val_dataset is not None:
                val_result = self._run_epoch(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    seed=None,
                    metric_fns=metric_fns,
                    training=False,
                )
                monitor_loss = val_result["loss"]
                history["val_loss"].append(monitor_loss)
                message += f" val_loss={monitor_loss:.4f}"
                for name in metric_fns:
                    value = val_result[name]
                    history[f"val_{name}"].append(value)
                    message += f" val_{name}={value:.4f}"

            message += f" elapsed={elapsed:.3f}s"
            if print_every is not None and (epoch == 1 or epoch % print_every == 0):
                print(message)

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    save_model(self.model, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                break

        return history

    def _resolve_metrics(self, metrics: Metrics) -> dict[str, MetricFn]:
        resolved: dict[str, MetricFn] = {}
        if self.metric_fn is not None:
            resolved["accuracy"] = self.metric_fn
        if metrics is not None:
            resolved.update(dict(metrics))
        return resolved

    def _run_epoch(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int | None,
        metric_fns: dict[str, MetricFn],
        training: bool,
    ) -> dict[str, float]:
        if hasattr(self.model, "train") and training:
            self.model.train()
        elif hasattr(self.model, "eval"):
            self.model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
        losses: list[float] = []
        metric_values: dict[str, list[float]] = {name: [] for name in metric_fns}

        for x_batch, y_batch in loader:
            predictions = self.model(x_batch)
            loss = self.loss_fn(predictions, y_batch)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
            for name, fn in metric_fns.items():
                metric_values[name].append(fn(predictions, y_batch))

        result = {"loss": float(np.mean(losses))}
        for name, values in metric_values.items():
            result[name] = float(np.mean(values))
        return result


def classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Metric helper for multi-class classification."""
    return accuracy(logits, labels)
