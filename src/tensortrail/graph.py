"""Computation graph export utilities for TensorTrail."""

from __future__ import annotations

from os import PathLike

from .tensor import Tensor


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _shape_label(tensor: Tensor) -> str:
    return "scalar" if tensor.shape == () else str(tensor.shape)


def visualize_graph(tensor: Tensor, path: str | PathLike[str] = "graph.dot") -> None:
    """Export the computation graph ending at ``tensor`` as Graphviz DOT.

    The DOT file is intentionally static and dependency-free. It can be opened
    as text or rendered later with the external Graphviz ``dot`` command.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("visualize_graph expects a Tensor.")

    nodes: list[Tensor] = []
    visited: set[Tensor] = set()

    def visit(node: Tensor) -> None:
        if node in visited:
            return
        visited.add(node)
        for parent in node._prev:
            visit(parent)
        nodes.append(node)

    visit(tensor)

    lines = [
        "digraph TensorTrail {",
        "  rankdir=LR;",
        '  node [shape=record, fontname="Menlo"];',
    ]
    for node in nodes:
        op = node._op or "leaf"
        label = (
            f"shape: {_shape_label(node)}|"
            f"op: {op}|"
            f"requires_grad: {node.requires_grad}"
        )
        lines.append(f'  n{id(node)} [label="{_dot_escape(label)}"];')

    for node in nodes:
        for parent in node._prev:
            lines.append(f"  n{id(parent)} -> n{id(node)};")

    lines.append("}")

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

