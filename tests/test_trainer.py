from tensortrail import CrossEntropyLoss, Linear, ReLU, Sequential
from tensortrail.data import make_mnist_like
from tensortrail.optim import Adam, SGD
from tensortrail.trainer import Trainer, classification_accuracy


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
