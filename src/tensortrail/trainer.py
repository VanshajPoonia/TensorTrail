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

_BUILTIN_METRICS: dict[str, MetricFn] = {
    "accuracy": lambda preds, labels: accuracy(preds, labels),
}


class Trainer:
    """Supervised training loop for TensorTrail models.

    Example::

        trainer = Trainer(
            model=model,
            loss_fn=CrossEntropyLoss(),
            optimizer=Adam(model.parameters(), lr=0.001),
            metrics=["accuracy"],
        )
        history = trainer.fit(train_loader, val_loader=val_loader, epochs=20, log_every=5)
        results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        metrics: list[str] | Metrics = None,
        metric_fn: MetricFn | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # ``metrics`` accepts a list of built-in names (e.g. ["accuracy"]) or a
        # mapping of {name: callable}.  ``metric_fn`` is the legacy parameter.
        self._metrics_arg = metrics
        self.metric_fn = metric_fn
        # Validate string metric names immediately so callers get fast feedback.
        if isinstance(metrics, list):
            for name in metrics:
                if name not in _BUILTIN_METRICS:
                    raise ValueError(
                        f"Unknown built-in metric {name!r}. "
                        f"Available: {sorted(_BUILTIN_METRICS)}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader | Dataset,
        val_loader: DataLoader | Dataset | None = None,
        epochs: int = 20,
        log_every: int | None = 1,
        early_stopping_patience: int | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        # ---- backward-compatible keyword arguments ----
        batch_size: int = 32,
        shuffle: bool = True,
        print_every: int | None = None,
        val_dataset: Dataset | None = None,
        metrics: Metrics = None,
    ) -> dict[str, list[float]]:
        """Train the model and return a history dictionary.

        Parameters
        ----------
        train_loader:
            A :class:`DataLoader` (preferred) or legacy :class:`Dataset`.
            When a ``Dataset`` is passed the ``batch_size`` and ``shuffle``
            keyword arguments control how it is batched.
        val_loader:
            Optional :class:`DataLoader` (or ``Dataset``) for validation.
        epochs:
            Number of full passes over the training data.
        log_every:
            Print a progress line after each *N* epochs (``None`` suppresses
            all output).  ``print_every`` takes precedence when set (legacy).
        early_stopping_patience:
            Stop training after this many consecutive epochs without a
            reduction in the monitored loss (validation loss when available,
            otherwise training loss).
        checkpoint_path:
            If set, save the best model weights to this path using
            :func:`save_model` whenever the monitored loss improves.
        batch_size:
            Batch size used when wrapping a ``Dataset`` (legacy).
        shuffle:
            Whether to shuffle each epoch when wrapping a ``Dataset`` (legacy).
        print_every:
            Legacy alias for ``log_every``; takes precedence when provided.
        val_dataset:
            Legacy keyword for passing a validation ``Dataset`` directly.
        metrics:
            Per-call override of metric functions (mapping of name → callable).
        """
        # --- normalise inputs ---
        train_loader = self._to_loader(train_loader, batch_size, shuffle, seed=42)

        if val_dataset is not None and val_loader is None:
            val_loader = val_dataset
        if val_loader is not None:
            val_loader = self._to_loader(val_loader, batch_size, shuffle=False, seed=None)

        # ``print_every`` wins for backward compatibility
        effective_log_every: int | None = print_every if print_every is not None else log_every

        metric_fns = self._resolve_metrics(metrics)

        # --- initialise history ---
        history: dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "elapsed_time": [],
        }
        history["loss"] = history["train_loss"]          # backward-compat alias
        if val_loader is not None:
            history["val_loss"] = []
        for name in metric_fns:
            history[name] = []
            if val_loader is not None:
                history[f"val_{name}"] = []
        if metric_fns:
            first_metric = next(iter(metric_fns))
            history["metric"] = history[first_metric]   # backward-compat alias

        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_result = self._run_loader(train_loader, metric_fns=metric_fns, training=True)
            elapsed = time.perf_counter() - start_time

            mean_loss = train_result["loss"]
            history["epoch"].append(float(epoch))
            history["train_loss"].append(mean_loss)
            history["elapsed_time"].append(elapsed)

            message = f"epoch {epoch:03d} train_loss={mean_loss:.4f}"
            for name in metric_fns:
                value = train_result[name]
                history[name].append(value)
                message += f" {name}={value:.4f}"

            monitor_loss = mean_loss
            if val_loader is not None:
                val_result = self._run_loader(val_loader, metric_fns=metric_fns, training=False)
                monitor_loss = val_result["loss"]
                history["val_loss"].append(monitor_loss)
                message += f" val_loss={monitor_loss:.4f}"
                for name in metric_fns:
                    value = val_result[name]
                    history[f"val_{name}"].append(value)
                    message += f" val_{name}={value:.4f}"

            message += f" elapsed={elapsed:.3f}s"
            if effective_log_every is not None and (
                epoch == 1 or epoch % effective_log_every == 0
            ):
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

    def evaluate(
        self,
        loader: DataLoader | Dataset,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Evaluate the model on *loader* and return a metrics dictionary.

        Returns a dict with at least ``"loss"`` plus any metrics configured
        on the Trainer (e.g. ``"accuracy"``).

        Parameters
        ----------
        loader:
            A :class:`DataLoader` or :class:`Dataset` to evaluate on.
        batch_size:
            Batch size used when wrapping a ``Dataset``.
        """
        loader = self._to_loader(loader, batch_size, shuffle=False, seed=None)
        metric_fns = self._resolve_metrics(None)
        return self._run_loader(loader, metric_fns=metric_fns, training=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_loader(
        self,
        loader_or_dataset: DataLoader | Dataset,
        batch_size: int,
        shuffle: bool,
        seed,
    ) -> DataLoader:
        if isinstance(loader_or_dataset, DataLoader):
            return loader_or_dataset
        if isinstance(loader_or_dataset, Dataset):
            return DataLoader(loader_or_dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
        raise TypeError(
            f"Expected DataLoader or Dataset, got {type(loader_or_dataset).__name__}"
        )

    def _resolve_metrics(self, extra_metrics: Metrics) -> dict[str, MetricFn]:
        resolved: dict[str, MetricFn] = {}

        # legacy single metric_fn
        if self.metric_fn is not None:
            resolved["accuracy"] = self.metric_fn

        # primary metrics argument (list of names or mapping)
        if self._metrics_arg is not None:
            if isinstance(self._metrics_arg, list):
                for name in self._metrics_arg:
                    if name not in _BUILTIN_METRICS:
                        raise ValueError(
                            f"Unknown built-in metric {name!r}. "
                            f"Available: {sorted(_BUILTIN_METRICS)}"
                        )
                    resolved[name] = _BUILTIN_METRICS[name]
            else:
                resolved.update(dict(self._metrics_arg))

        # per-call override
        if extra_metrics is not None:
            resolved.update(dict(extra_metrics))

        return resolved

    def _run_loader(
        self,
        loader: DataLoader,
        *,
        metric_fns: dict[str, MetricFn],
        training: bool,
    ) -> dict[str, float]:
        if training and hasattr(self.model, "train"):
            self.model.train()
        elif hasattr(self.model, "eval"):
            self.model.eval()

        losses: list[float] = []
        metric_values: dict[str, list[float]] = {name: [] for name in metric_fns}

        for x_batch, y_batch in loader:
            predictions = self.model(x_batch)
            loss = self.loss_fn(predictions, y_batch)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            losses.append(float(loss.item()))
            for name, fn in metric_fns.items():
                metric_values[name].append(fn(predictions, y_batch))

        result: dict[str, float] = {"loss": float(np.mean(losses))}
        for name, values in metric_values.items():
            result[name] = float(np.mean(values))
        return result


def classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Metric helper for multi-class classification."""
    return accuracy(logits, labels)
