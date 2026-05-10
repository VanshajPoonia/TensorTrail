"""Train a TensorTrail MLP on a synthetic MNIST-like classification task."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import (
    Adam,
    CrossEntropyLoss,
    Linear,
    ReLU,
    Sequential,
    Tensor,
    Trainer,
    classification_accuracy,
    make_mnist_like,
    train_test_split,
)


def main() -> None:
    dataset = make_mnist_like(n_samples=800, n_features=64, n_classes=10, seed=11)
    train_data, test_data = train_test_split(dataset.x, dataset.y, test_size=0.25, seed=11)

    model = Sequential(
        Linear(64, 32, seed=10),
        ReLU(),
        Linear(32, 10, seed=20),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.03)
    trainer = Trainer(model, loss_fn, optimizer, metric_fn=classification_accuracy)

    trainer.fit(train_data, epochs=40, batch_size=64, shuffle=True, print_every=5)

    test_logits = model(Tensor(test_data.x))
    test_acc = classification_accuracy(test_logits, Tensor(test_data.y))
    print(f"\nfinal test accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
