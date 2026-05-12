"""Tests for data.py — Dataset, DataLoader, and helpers."""

import numpy as np
import pytest

from tensortrail.data import DataLoader, Dataset, make_mnist_like, train_test_split
from tensortrail.tensor import Tensor


def _make_ds(n=50, n_feat=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_feat))
    y = rng.integers(0, 3, size=n)
    return Dataset(x, y)


# ------------------------------------------------------------------ #
# Dataset                                                             #
# ------------------------------------------------------------------ #

def test_dataset_len_and_getitem():
    ds = _make_ds(30, 4)
    assert len(ds) == 30
    x_i, y_i = ds[0]
    assert x_i.shape == (4,)
    assert y_i.shape == ()


def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Dataset(np.zeros((10, 4)), np.zeros(9))


# ------------------------------------------------------------------ #
# DataLoader                                                          #
# ------------------------------------------------------------------ #

def test_dataloader_returns_tensor_batches():
    ds = _make_ds(20, 4)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3          # ceil(20/8)
    x_b, y_b = batches[0]
    assert isinstance(x_b, Tensor)
    assert isinstance(y_b, Tensor)
    assert x_b.shape == (8, 4)
    assert y_b.shape == (8,)


def test_dataloader_rejects_non_positive_batch_size():
    ds = _make_ds(20, 4)
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(ds, batch_size=0)


def test_dataloader_last_batch_smaller():
    ds = _make_ds(25, 4)
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    # last batch has only 5 samples
    assert batches[-1][0].shape[0] == 5


def test_dataloader_covers_all_samples_without_shuffle():
    ds = _make_ds(30, 4)
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    collected_x = np.vstack([x.data for x, _ in loader])
    np.testing.assert_array_equal(collected_x, ds.x)


def test_dataloader_shuffle_changes_order():
    ds = _make_ds(40, 4)
    loader_a = DataLoader(ds, batch_size=40, shuffle=True, seed=1)
    loader_b = DataLoader(ds, batch_size=40, shuffle=True, seed=2)
    x_a = list(loader_a)[0][0].data
    x_b = list(loader_b)[0][0].data
    # Different seeds → different ordering
    assert not np.array_equal(x_a, x_b)


def test_dataloader_no_shuffle_is_deterministic():
    ds = _make_ds(20, 4)
    loader = DataLoader(ds, batch_size=20, shuffle=False)
    x1 = list(loader)[0][0].data
    x2 = list(loader)[0][0].data
    np.testing.assert_array_equal(x1, x2)


def test_dataloader_works_for_classification_labels():
    ds = _make_ds(16, 4)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    for x_batch, y_batch in loader:
        assert x_batch.data.dtype == float
        assert y_batch.shape[0] == 8


# ------------------------------------------------------------------ #
# train_test_split                                                     #
# ------------------------------------------------------------------ #

def test_train_test_split_sizes():
    x = np.zeros((100, 4))
    y = np.arange(100)
    train_ds, test_ds = train_test_split(x, y, test_size=0.2, seed=0)
    assert len(train_ds) == 80
    assert len(test_ds) == 20


def test_train_test_split_no_overlap():
    x = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50)
    train_ds, test_ds = train_test_split(x, y, test_size=0.2, seed=0)
    train_vals = set(train_ds.y.tolist())
    test_vals = set(test_ds.y.tolist())
    assert train_vals.isdisjoint(test_vals)
    assert len(train_vals) + len(test_vals) == 50


def test_train_test_split_invalid_size():
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 2)), np.zeros(10), test_size=1.5)


# ------------------------------------------------------------------ #
# make_mnist_like                                                      #
# ------------------------------------------------------------------ #

def test_make_mnist_like_shapes():
    ds = make_mnist_like(n_samples=60, n_features=16, n_classes=5, seed=3)
    assert ds.x.shape == (60, 16)
    assert ds.y.shape == (60,)
    assert set(np.unique(ds.y)).issubset(set(range(5)))
