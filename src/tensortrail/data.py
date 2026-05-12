"""Small offline data utilities for TensorTrail."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .tensor import Tensor


class Dataset:
    """Simple in-memory dataset of feature and target arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y)
        if len(self.x) != len(self.y):
            raise ValueError("features and targets must have the same length.")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x[index], self.y[index]


class DataLoader:
    """Mini-batch iterator over a Dataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("DataLoader batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            x_batch, y_batch = self.dataset[batch_idx]
            yield Tensor(x_batch), Tensor(y_batch)


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Split arrays into train and test Dataset objects."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)
    test_count = int(round(len(indices) * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return Dataset(x[train_idx], y[train_idx]), Dataset(x[test_idx], y[test_idx])


def make_xor(repeats: int = 128, noise: float = 0.05, seed: int | None = 42) -> Dataset:
    """Create an offline XOR dataset."""
    rng = np.random.default_rng(seed)
    base_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    base_y = np.array([[0], [1], [1], [0]], dtype=float)
    x = np.tile(base_x, (repeats, 1))
    y = np.tile(base_y, (repeats, 1))
    if noise:
        x = x + rng.normal(0, noise, size=x.shape)
    return Dataset(x, y)


def make_mnist_like(
    n_samples: int = 600,
    n_features: int = 64,
    n_classes: int = 10,
    seed: int | None = 7,
) -> Dataset:
    """Generate a small synthetic flattened-image classification dataset.

    Each class owns a smooth prototype vector. Samples are noisy versions of
    their class prototype, clipped to an image-like [0, 1] range.
    """
    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.1, 0.9, size=(n_classes, n_features))
    labels = rng.integers(0, n_classes, size=n_samples)
    x = prototypes[labels] + rng.normal(0, 0.12, size=(n_samples, n_features))
    x = np.clip(x, 0, 1)
    return Dataset(x, labels)
