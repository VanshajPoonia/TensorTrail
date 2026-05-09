"""Train a tiny TensorTrail CNN on synthetic image-like data."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import (  # noqa: E402
    Conv2D,
    CrossEntropyLoss,
    DataLoader,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
)
from tensortrail.data import Dataset, train_test_split  # noqa: E402
from tensortrail.optim import Adam  # noqa: E402
from tensortrail.trainer import Trainer  # noqa: E402


def make_tiny_image_dataset(
    n_samples: int = 360,
    image_size: int = 8,
    noise: float = 0.08,
    seed: int = 11,
) -> Dataset:
    """Create small NCHW images with class-specific bright patterns."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, 1, image_size, image_size), dtype=float)
    labels = rng.integers(0, 3, size=n_samples)

    for index, label in enumerate(labels):
        image = np.zeros((image_size, image_size), dtype=float)
        if label == 0:
            image[:, image_size // 2 - 1 : image_size // 2 + 1] = 1.0
        elif label == 1:
            image[image_size // 2 - 1 : image_size // 2 + 1, :] = 1.0
        else:
            np.fill_diagonal(image, 1.0)
            np.fill_diagonal(np.fliplr(image), 0.6)

        image += rng.normal(0.0, noise, size=image.shape)
        images[index, 0] = np.clip(image, 0.0, 1.0)

    return Dataset(images, labels)


def main() -> None:
    dataset = make_tiny_image_dataset()
    train_data, val_data = train_test_split(dataset.x, dataset.y, test_size=0.25, seed=7)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, seed=3)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = Sequential(
        Conv2D(1, 4, kernel_size=3, padding=1, seed=1),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(4 * 4 * 4, 3, seed=2),
    )
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.03),
        metrics=["accuracy"],
    )

    print(f"train={len(train_data)} samples  val={len(val_data)} samples")
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=20,
        log_every=5,
        early_stopping_patience=6,
    )
    results = trainer.evaluate(val_loader)

    print(
        "\ntiny CNN finished"
        f"\n  epochs run      : {len(history['epoch'])}"
        f"\n  final train_loss: {history['train_loss'][-1]:.4f}"
        f"\n  final val_loss  : {history['val_loss'][-1]:.4f}"
        f"\n  final val_acc   : {history['val_accuracy'][-1]:.4f}"
        f"\n  eval accuracy   : {results['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
