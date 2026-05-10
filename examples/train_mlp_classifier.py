"""Train a TensorTrail MLP on an offline synthetic multi-class dataset.

Demonstrates the full Trainer API:
  - DataLoader batching with shuffle
  - metrics=["accuracy"] in the Trainer constructor
  - val_loader for per-epoch validation
  - log_every progress printing
  - early_stopping_patience
  - trainer.evaluate() on a held-out test set
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import (
    Adam,
    BatchNorm1D,
    CrossEntropyLoss,
    DataLoader,
    Dropout,
    Linear,
    ReLU,
    Sequential,
    Trainer,
    make_mnist_like,
    train_test_split,
)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Data — synthetic 4-class dataset with 32 features per sample     #
    # ------------------------------------------------------------------ #
    dataset = make_mnist_like(n_samples=1200, n_features=32, n_classes=4, seed=21)

    # 70% train / 15% val / 15% test
    trainval_data, test_data = train_test_split(
        dataset.x, dataset.y, test_size=0.15, seed=21
    )
    train_data, val_data = train_test_split(
        trainval_data.x, trainval_data.y, test_size=0.15 / 0.85, seed=22
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, seed=42)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    print(
        f"dataset sizes  train={len(train_data)}  val={len(val_data)}"
        f"  test={len(test_data)}"
    )

    # ------------------------------------------------------------------ #
    # 2. Model                                                             #
    # ------------------------------------------------------------------ #
    model = Sequential(
        Linear(32, 24, seed=1),
        BatchNorm1D(24),
        ReLU(),
        Dropout(p=0.15, seed=2),
        Linear(24, 4, seed=3),
    )

    # ------------------------------------------------------------------ #
    # 3. Trainer — new DataLoader-first API                               #
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.025),
        metrics=["accuracy"],
    )

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=40,
        log_every=5,
        early_stopping_patience=8,
    )

    # ------------------------------------------------------------------ #
    # 4. Results                                                           #
    # ------------------------------------------------------------------ #
    print(
        "\ntraining complete"
        f"\n  epochs run     : {len(history['epoch'])}"
        f"\n  final train_loss: {history['train_loss'][-1]:.4f}"
        f"\n  final val_loss  : {history['val_loss'][-1]:.4f}"
        f"\n  final val_acc   : {history['val_accuracy'][-1]:.4f}"
    )

    results = trainer.evaluate(test_loader)
    print(
        f"\ntest set  loss={results['loss']:.4f}  accuracy={results['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
