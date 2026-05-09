# Autograd Explained

TensorTrail uses reverse-mode automatic differentiation. The design is small on
purpose: every operation builds a little graph node, and every graph node knows
how to send gradients back to its parents.

## Tensor

`Tensor` wraps a NumPy array and adds autograd state:

- `data`: the numeric NumPy array.
- `requires_grad`: whether this tensor should receive gradients.
- `grad`: accumulated gradient after `backward()`.
- `_prev`: parent tensors used to create this tensor.
- `_op`: a short operation name such as `add`, `matmul`, `relu`, or `conv2d`.
- `_backward`: a closure that applies the local derivative rule.

Leaf tensors usually come from user data or model parameters. Intermediate
tensors are created by operations such as addition, matrix multiplication,
reductions, activations, convolution, and pooling.

## Computational Graph

TensorTrail builds a dynamic graph while Python code runs. For example:

```python
y = ((x * x).tanh()).sum()
```

This creates graph nodes for multiplication, tanh, and sum. The final scalar
`y` points backward to its parents, and those parents point back until the graph
reaches the leaf tensor `x`.

Dynamic graph construction keeps the implementation easy to inspect: normal
Python control flow creates normal TensorTrail graphs.

## Chain Rule

The chain rule is the engine behind backpropagation. If:

```python
z = x * w
loss = z.sum()
```

then the multiply operation knows the local derivatives:

```text
dz/dx = w
dz/dw = x
```

During backpropagation, the multiply node receives `dLoss/dz` from later in the
graph and contributes:

```text
dLoss/dx = dLoss/dz * w
dLoss/dw = dLoss/dz * x
```

TensorTrail implements one local derivative rule per operation.

## Backward Closures

Each differentiable operation attaches a `_backward` closure to its output
tensor. The closure captures the parent tensors and any forward-pass values
needed for gradients.

For example, `tanh` stores the forward result:

```text
d/dx tanh(x) = 1 - tanh(x)^2
```

The graph traversal code does not need to know the derivative of every
operation. It only calls each node's `_backward` closure in the correct order.

## Topological Sorting

Backpropagation must run after all downstream gradient contributions have been
collected. TensorTrail handles this with a depth-first traversal from the final
tensor.

The traversal records parents before children, producing a topological order.
`backward()` then walks that list in reverse so every node sees the gradient of
the final output with respect to itself before it pushes gradients to parents.

## Gradient Accumulation

A tensor can feed multiple graph branches:

```python
y = x * x + x
```

The same `x` contributes through both the multiplication branch and the addition
branch. TensorTrail uses `_add_grad()` to accumulate each contribution into
`x.grad` instead of overwriting previous contributions.

This is essential for shared parameters, reused activations, and any graph where
one tensor influences the output through multiple paths.

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

Without this step, bias gradients and normalization parameter gradients would
have the wrong shape.

## Non-Scalar Backward

Calling `backward()` without an argument is only valid for scalar outputs.
Non-scalar tensors require an external gradient with the same shape:

```python
y.backward(np.ones_like(y.data))
```

This mirrors the mathematical idea that backpropagation starts from a seed
gradient.

## Why This Design Is Educational

TensorTrail keeps autograd close to the operations themselves. The tradeoff is
that it is not optimized for speed, but the benefit is clarity: each derivative
rule can be read, tested, and compared against finite differences.
