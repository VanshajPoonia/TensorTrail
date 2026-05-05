from tensortrail import Tensor, visualize_graph


def test_visualize_graph_writes_dot_file(tmp_path):
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = ((x * x).sum() + x.mean()).tanh()
    path = tmp_path / "graph.dot"

    visualize_graph(y, path)

    dot = path.read_text()
    assert "digraph TensorTrail" in dot
    assert "shape:" in dot
    assert "requires_grad: True" in dot
    assert "op: mul" in dot
    assert "op: sum" in dot
    assert "op: tanh" in dot

