import re

from tensortrail import Tensor, visualize_graph


def test_visualize_graph_writes_dot_file(tmp_path):
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = ((x * x).sum() + x.mean()).sigmoid().tanh()
    path = tmp_path / "graph.dot"

    visualize_graph(y, path)

    dot = path.read_text()
    assert dot.startswith("digraph TensorTrail")
    assert "  n0 [label=" in dot
    assert "shape:" in dot
    assert "requires_grad: True" in dot
    assert "op: mul" in dot
    assert "op: sum" in dot
    assert "op: mean" in dot
    assert "op: sigmoid" in dot
    assert "op: tanh" in dot
    assert not re.search(r"\bn\d{6,}\b", dot)
    assert dot.count(" -> ") == len(set(line for line in dot.splitlines() if " -> " in line))
