# Autograd Explained

TensorTrail uses reverse-mode automatic differentiation. That is the same
high-level idea behind large deep learning frameworks, but this version is small
enough to read in one sitting.

## Tensor Object

`Tensor` wraps a NumPy array and adds three pieces of autograd state:

- `requires_grad`, which says whether operations should be tracked.
- `grad`, which stores accumulated gradients after `backward()`.
- `_prev`, `_op`, and `_backward`, which describe how this tensor was created.

Leaf tensors usually come from user data or model parameters. Intermediate
tensors come from operations such as addition, matrix multiplication, reductions,
and activations.

## Computational Graph

Every differentiable operation creates a new output `Tensor`. If any input
requires gradients, the output stores references to its parent tensors. For
example:

```python
y = ((x * x).tanh()).sum()
```

creates a graph with multiply, tanh, and sum nodes. The final scalar `y` points
back through that graph to `x`.

## Topological Sort

Backpropagation has to run in dependency order. TensorTrail first walks backward
from the final tensor with depth-first search and records a topological ordering:
parents before children.

Then it reverses that list. This means each node's gradient has already been
collected by the time its local backward function runs.

## Chain Rule

The chain rule combines local derivatives into full derivatives. If:

```python
z = x * y
loss = z.sum()
```

then the multiply operation knows:

- `dz/dx = y`
- `dz/dy = x`

During backpropagation it receives `dLoss/dz` and contributes:

- `dLoss/dx = dLoss/dz * y`
- `dLoss/dy = dLoss/dz * x`

TensorTrail implements one small local derivative rule per operation.

## Backward Closures

Each operation attaches a `_backward` closure to its output tensor. The closure
captures the input tensors and any values needed to compute derivatives.

For `tanh`, the forward pass computes `tanh(x)`. The backward closure reuses
that value:

```text
d/dx tanh(x) = 1 - tanh(x)^2
```

The graph traversal code stays generic; each operation owns its own gradient
recipe.

## Gradient Accumulation

A tensor can feed multiple graph branches:

```python
y = x * x + x
```

The same `x` contributes through both the multiply branch and the addition
branch. TensorTrail adds each contribution into `x.grad`. This is why model
parameters can be reused across many examples in a batch and still receive the
correct total gradient.

## Broadcasting Gradients

NumPy broadcasting expands smaller arrays during the forward pass:

```python
x.shape == (4, 3)
b.shape == (3,)
y = x + b
```

The bias `b` is used once per row. In the backward pass, its gradient must be
summed back down to shape `(3,)`. TensorTrail's unbroadcast helper removes extra
leading dimensions and sums axes where the original input had size `1`.

Without this step, gradients for biases and BatchNorm parameters would have the
wrong shape.

## Why Gradient Checking Matters

Autograd code can look correct while hiding small shape or sign bugs. Gradient
checking compares TensorTrail's analytical gradients with numerical finite
differences:

```text
df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)
```

It is too slow for training, but excellent for testing new operations. If a
smooth scalar-valued function passes gradient checking, the local backward rules
used by that function are much more trustworthy.

