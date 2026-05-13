"""Plot TensorTrail Trainer history for an offline synthetic dataset.

This example is optional because plotting uses Matplotlib. Install it with:

    python -m pip install -e ".[plot]"
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR", str(pathlib.Path(tempfile.gettempdir()) / "tensortrail-mpl")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        'Matplotlib is required for this optional example. '
        'Install it with: python -m pip install -e ".[plot]"'
    ) from exc

from tensortrail import (
    Adam,
    CrossEntropyLoss,
    DataLoader,
    Linear,
    ReLU,
    Sequential,
    Trainer,
    make_mnist_like,
    train_test_split,
)


def main() -> None:
    dataset = make_mnist_like(n_samples=500, n_features=16, n_classes=3, seed=31)
    train_data, val_data = train_test_split(dataset.x, dataset.y, test_size=0.25, seed=31)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, seed=10)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = Sequential(
        Linear(16, 12, seed=1),
        ReLU(),
        Linear(12, 3, seed=2),
    )
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.02),
        metrics=["accuracy"],
        clip_grad_norm=5.0,
    )

    history = trainer.fit(train_loader, val_loader=val_loader, epochs=20, log_every=None)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    output_path = ROOT / "training_history.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path.name}")


if __name__ == "__main__":
    main()
