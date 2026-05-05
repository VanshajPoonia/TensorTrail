"""Export a small TensorTrail autograd graph to Graphviz DOT."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensortrail import Tensor, visualize_graph


def main() -> None:
    x = Tensor([[1.0, -2.0], [3.0, 0.5]], requires_grad=True)
    y = ((x * x).tanh() + x.exp()).mean()
    y.backward()

    output_path = ROOT / "graph.dot"
    visualize_graph(y, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

