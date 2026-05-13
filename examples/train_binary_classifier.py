"""Train a small TensorTrail binary classifier from raw logits."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import Adam, BCEWithLogitsLoss, Linear, ReLU, Sequential, Tensor


def make_binary_dataset(n_samples: int = 300, seed: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    negative = rng.normal(loc=(-1.2, -1.0), scale=0.45, size=(half, 2))
    positive = rng.normal(loc=(1.2, 1.0), scale=0.45, size=(n_samples - half, 2))
    x = np.vstack([negative, positive])
    y = np.vstack([np.zeros((half, 1)), np.ones((n_samples - half, 1))])
    indices = rng.permutation(n_samples)
    return x[indices], y[indices]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0,
        1 / (1 + np.exp(-values)),
        np.exp(values) / (1 + np.exp(values)),
    )


def main() -> None:
    x_data, y_data = make_binary_dataset()
    x = Tensor(x_data)
    y = Tensor(y_data)

    model = Sequential(
        Linear(2, 8, seed=1),
        ReLU(),
        Linear(8, 1, seed=2),
    )
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    for epoch in range(1, 301):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 75 == 0:
            probabilities = sigmoid(logits.data)
            predictions = (probabilities >= 0.5).astype(float)
            accuracy = float(np.mean(predictions == y.data))
            print(f"epoch {epoch:03d} loss={loss.item():.4f} accuracy={accuracy:.4f}")

    final_probabilities = sigmoid(model(x).data)
    final_predictions = (final_probabilities >= 0.5).astype(float)
    final_accuracy = float(np.mean(final_predictions == y.data))
    print(f"\nfinal accuracy={final_accuracy:.4f}")


if __name__ == "__main__":
    main()
