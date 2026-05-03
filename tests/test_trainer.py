from tensortrail import CrossEntropyLoss, Linear, ReLU, Sequential
from tensortrail.data import make_mnist_like
from tensortrail.optim import Adam
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

