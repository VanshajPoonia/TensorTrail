"""Train a regularized TensorTrail MLP on a synthetic classification task.

Showcases the regularization layers:
  - BatchNorm1D  — normalizes across the batch per feature
  - Dropout      — randomly zeros activations during training

Both layers behave differently in train vs eval mode; the Trainer handles
mode switching automatically.
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
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    dataset = make_mnist_like(n_samples=1000, n_features=32, n_classes=4, seed=7)
    train_data, val_data = train_test_split(dataset.x, dataset.y, test_size=0.2, seed=7)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, seed=1)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print(f"train={len(train_data)} samples  val={len(val_data)} samples")

    # ------------------------------------------------------------------ #
    # Regularized two-hidden-layer MLP                                     #
    #                                                                      #
    #   Linear → BatchNorm1D → ReLU → Dropout                             #
    #   → Linear → BatchNorm1D → ReLU → Dropout                           #
    #   → Linear (output logits)                                           #
    # ------------------------------------------------------------------ #
    model = Sequential(
        Linear(32, 64, seed=1),
        BatchNorm1D(64),
        ReLU(),
        Dropout(p=0.2, seed=2),
        Linear(64, 32, seed=3),
        BatchNorm1D(32),
        ReLU(),
        Dropout(p=0.2, seed=4),
        Linear(32, 4, seed=5),
    )

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.01),
        metrics=["accuracy"],
    )

    print("\ntraining (log_every=5, early_stopping_patience=8):\n")
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=50,
        log_every=5,
        early_stopping_patience=8,
    )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    n_epochs = len(history["epoch"])
    print(
        f"\ntraining finished after {n_epochs} epochs"
        f"\n  final train_loss : {history['train_loss'][-1]:.4f}"
        f"\n  final train_acc  : {history['accuracy'][-1]:.4f}"
        f"\n  final val_loss   : {history['val_loss'][-1]:.4f}"
        f"\n  final val_acc    : {history['val_accuracy'][-1]:.4f}"
    )

    results = trainer.evaluate(val_loader)
    print(
        f"\neval on val set → loss={results['loss']:.4f}"
        f"  accuracy={results['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
