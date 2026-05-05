"""Train a TensorTrail MLP on an offline synthetic classification dataset."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import BatchNorm1D, CrossEntropyLoss, Dropout, Linear, ReLU, Sequential
from tensortrail.data import make_mnist_like, train_test_split
from tensortrail.optim import Adam
from tensortrail.trainer import Trainer, classification_accuracy


def main() -> None:
    dataset = make_mnist_like(n_samples=900, n_features=32, n_classes=4, seed=21)
    train_data, val_data = train_test_split(dataset.x, dataset.y, test_size=0.25, seed=21)

    model = Sequential(
        Linear(32, 24, seed=1),
        BatchNorm1D(24),
        ReLU(),
        Dropout(p=0.15, seed=2),
        Linear(24, 4, seed=3),
    )
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), lr=0.025),
        metric_fn=classification_accuracy,
    )

    history = trainer.fit(
        train_data,
        val_dataset=val_data,
        epochs=35,
        batch_size=64,
        shuffle=True,
        print_every=5,
        early_stopping_patience=8,
    )

    print(
        "\nfinal "
        f"train_loss={history['train_loss'][-1]:.4f} "
        f"val_loss={history['val_loss'][-1]:.4f} "
        f"val_accuracy={history['val_accuracy'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()

