import pytest

from tensortrail import (
    Adam,
    CrossEntropyLoss,
    DataLoader,
    Dataset,
    Linear,
    ReLU,
    SGD,
    Sequential,
    Trainer,
    classification_accuracy,
    make_mnist_like,
)


# ------------------------------------------------------------------ #
# Legacy API (Dataset + batch_size) — must remain working             #
# ------------------------------------------------------------------ #

def test_trainer_records_richer_history_with_backward_compatible_aliases():
    dataset = make_mnist_like(n_samples=80, n_features=8, n_classes=3, seed=5)
    model = Sequential(Linear(8, 6, seed=1), ReLU(), Linear(6, 3, seed=2))
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), lr=0.02),
        metric_fn=classification_accuracy,
    )

    history = trainer.fit(dataset, epochs=3, batch_size=16, shuffle=False)

    assert history["epoch"] == [1.0, 2.0, 3.0]
    assert len(history["train_loss"]) == 3
    assert len(history["accuracy"]) == 3
    assert len(history["elapsed_time"]) == 3
    assert history["loss"] is history["train_loss"]
    assert history["metric"] is history["accuracy"]
    assert all(elapsed >= 0 for elapsed in history["elapsed_time"])


def test_trainer_validation_early_stopping_and_checkpoint(tmp_path):
    dataset = make_mnist_like(n_samples=90, n_features=6, n_classes=3, seed=8)
    train_data = type(dataset)(dataset.x[:60], dataset.y[:60])
    val_data = type(dataset)(dataset.x[60:], dataset.y[60:])
    model = Sequential(Linear(6, 5, seed=1), ReLU(), Linear(5, 3, seed=2))
    checkpoint_path = tmp_path / "best_model.npz"
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        SGD(model.parameters(), lr=0.0),
        metric_fn=classification_accuracy,
    )

    history = trainer.fit(
        train_data,
        val_dataset=val_data,
        epochs=10,
        batch_size=15,
        shuffle=False,
        early_stopping_patience=2,
        checkpoint_path=checkpoint_path,
    )

    assert len(history["epoch"]) == 3
    assert len(history["val_loss"]) == 3
    assert len(history["val_accuracy"]) == 3
    assert checkpoint_path.exists()


# ------------------------------------------------------------------ #
# New DataLoader-first API                                             #
# ------------------------------------------------------------------ #

def test_trainer_fit_with_dataloader_returns_history():
    dataset = make_mnist_like(n_samples=80, n_features=8, n_classes=3, seed=5)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = Sequential(Linear(8, 6, seed=1), ReLU(), Linear(6, 3, seed=2))
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.02),
        metrics=["accuracy"],
    )

    history = trainer.fit(train_loader, epochs=3, log_every=None)

    assert history["epoch"] == [1.0, 2.0, 3.0]
    assert len(history["train_loss"]) == 3
    assert len(history["accuracy"]) == 3
    assert len(history["elapsed_time"]) == 3
    assert history["loss"] is history["train_loss"]
    assert history["metric"] is history["accuracy"]


def test_trainer_evaluate_returns_metrics():
    dataset = make_mnist_like(n_samples=60, n_features=8, n_classes=3, seed=7)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    model = Sequential(Linear(8, 6, seed=1), ReLU(), Linear(6, 3, seed=2))
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.0),
        metrics=["accuracy"],
    )

    results = trainer.evaluate(loader)

    assert "loss" in results
    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert results["loss"] >= 0.0


def test_trainer_early_stopping_with_dataloader():
    dataset = make_mnist_like(n_samples=60, n_features=6, n_classes=3, seed=9)
    train_ds = Dataset(dataset.x[:40], dataset.y[:40])
    val_ds = Dataset(dataset.x[40:], dataset.y[40:])
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False)
    model = Sequential(Linear(6, 5, seed=1), ReLU(), Linear(5, 3, seed=2))
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=SGD(model.parameters(), lr=0.0),  # loss stays constant
        metrics=["accuracy"],
    )

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=10,
        log_every=None,
        early_stopping_patience=2,
    )

    # With lr=0.0, loss never improves after epoch 1.
    # patience=2 → stops at epoch 3.
    assert len(history["epoch"]) == 3
    assert len(history["val_loss"]) == 3


def test_trainer_checkpoint_saving_with_dataloader(tmp_path):
    dataset = make_mnist_like(n_samples=40, n_features=6, n_classes=3, seed=11)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = Sequential(Linear(6, 5, seed=1), ReLU(), Linear(5, 3, seed=2))
    checkpoint_path = tmp_path / "best.npz"
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.05),
    )

    trainer.fit(loader, epochs=3, log_every=None, checkpoint_path=checkpoint_path)

    assert checkpoint_path.exists()


def test_trainer_unknown_metric_raises():
    model = Sequential(Linear(4, 3, seed=1))
    with pytest.raises(ValueError, match="Unknown built-in metric"):
        Trainer(model=model, loss_fn=None, optimizer=None, metrics=["f1_score"])
