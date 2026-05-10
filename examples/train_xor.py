"""Train a tiny TensorTrail network on XOR."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import Adam, BinaryCrossEntropyLoss, Linear, Sequential, Sigmoid, Tanh, Tensor, make_xor


def main() -> None:
    np.random.seed(3)
    dataset = make_xor(repeats=64, noise=0.02, seed=3)

    model = Sequential(
        Linear(2, 8, seed=1),
        Tanh(),
        Linear(8, 1, seed=2),
        Sigmoid(),
    )
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    x = Tensor(dataset.x)
    y = Tensor(dataset.y)

    for epoch in range(1, 1001):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch:04d} loss={loss.item():.4f}")

    clean_x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float))
    clean_predictions = model(clean_x).data
    print("\nFinal XOR predictions:")
    for inputs, pred in zip(clean_x.data.astype(int), clean_predictions):
        print(f"{inputs.tolist()} -> {pred[0]:.3f}")


if __name__ == "__main__":
    main()

