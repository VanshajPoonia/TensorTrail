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
## Chain Rule

The chain rule says that if one value depends on another through intermediate steps, gradients multiply along that path.

For multiplication:

```python
z = x * y
```

the local derivatives are:

- `dz/dx = y`
- `dz/dy = x`

So the backward closure receives `dL/dz` and contributes:

- `dL/dx = dL/dz * y`
- `dL/dy = dL/dz * x`

TensorTrail encodes one of these small rules for every differentiable tensor operation.

## Backward Closures

A backward closure is the operation-specific gradient recipe captured during the forward pass. It closes over:

- the input tensors
- the output tensor
- any values needed for the derivative

For `tanh`, the forward pass stores the computed `tanh(x)` value. The backward closure reuses it to apply:

```text
d/dx tanh(x) = 1 - tanh(x)^2
```

This keeps the autograd engine generic: graph traversal is shared, while each operation owns its local derivative.
