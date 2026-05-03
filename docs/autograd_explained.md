# Autograd Explained

TensorTrail uses reverse-mode automatic differentiation, the same core idea that powers large machine learning frameworks. The implementation is intentionally small so the path from forward computation to gradients is visible.
## Computational Graph

Every `Tensor` stores NumPy data and, when gradients are required, references to the tensors that created it. For example:

```python
z = (x * y + x).sum()
```

creates a small graph where `z` depends on addition, multiplication, and sum nodes, which ultimately depend on `x` and `y`.

Each operation also stores a backward closure: a tiny function that knows how to pass the output gradient back to that operation's inputs.

## Topological Sorting

Backpropagation must run from the final output back toward the original inputs. TensorTrail first walks the graph with depth-first search and records nodes in topological order, where parents appear before children.

Then `backward()` reverses that order. This guarantees that when a node's backward closure runs, the gradient flowing into that node has already been accumulated.
