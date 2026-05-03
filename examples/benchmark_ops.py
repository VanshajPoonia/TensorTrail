"""Educational operation benchmark: TensorTrail versus raw NumPy."""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import Tensor


def numpy_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def time_it(label: str, fn, repeats: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    per_run_ms = elapsed / repeats * 1000
    print(f"{label:<28} {per_run_ms:8.4f} ms/run")
    return per_run_ms


def main() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=(128, 128))
    b = rng.normal(size=(128, 128))
    x_np = rng.normal(size=(1024, 64))
    x_tt = Tensor(x_np)
    a_tt = Tensor(a)
    b_tt = Tensor(b)
    a_grad = Tensor(a, requires_grad=True)
    b_grad = Tensor(b, requires_grad=True)

    print("TensorTrail operation benchmark")
    print("Educational only: TensorTrail adds Python/autograd overhead, so NumPy should win.")
    print()

    time_it("NumPy add + tanh", lambda: np.tanh(x_np + 2.0))
    time_it("TensorTrail add + tanh", lambda: (x_tt + 2.0).tanh().data)
    print()
    time_it("NumPy matmul", lambda: numpy_matmul(a, b), repeats=100)
    time_it("TensorTrail matmul", lambda: (a_tt @ b_tt).data, repeats=100)
    print()


if __name__ == "__main__":
    main()
