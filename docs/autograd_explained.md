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
## Gradient Accumulation

A tensor can be used in more than one branch of a graph:

```python
y = x * x + x
```

Here `x` contributes through both `x * x` and `+ x`. During backpropagation, TensorTrail adds each contribution into `x.grad`. This accumulation is what makes shared parameters and branching computation graphs work.

## Broadcasting Gradients

NumPy broadcasting lets tensors of different shapes participate in one operation:

```python
x.shape == (4, 3)
b.shape == (3,)
y = x + b
```

The forward pass stretches `b` across the batch dimension. In the backward pass, the gradient for `b` must be reduced back to shape `(3,)`, summing over the broadcasted dimension.

TensorTrail uses an internal unbroadcast helper to:

- remove extra leading dimensions
- sum over axes where the original operand had size `1`
- reshape the gradient back to the operand's original shape

Without this step, gradients for biases and other broadcasted tensors would have the wrong shape.
