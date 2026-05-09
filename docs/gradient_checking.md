# Gradient Checking

Gradient checking is a debugging technique for autograd systems. It answers a
simple question: do the gradients computed by `backward()` match the slope of
the function measured numerically?

## Why It Matters

Backward rules can be wrong in subtle ways:

- a missing transpose
- an incorrect broadcast reduction
- a sign error
- a wrong axis in a reduction
- a shape bug that only appears for batched inputs

Training may still appear to run with small gradient bugs, but optimization can
be slower, unstable, or silently incorrect. Gradient checking gives each new
operation a direct correctness test.

## Finite Differences

For a scalar-valued function `f`, the centered finite-difference estimate is:

```text
df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)
```

TensorTrail perturbs one input element at a time, evaluates the function at the
positive and negative perturbations, and stores the estimated numerical
gradient.

Centered differences are more accurate than one-sided differences, but still
depend on a good `eps`. Too large and the estimate is coarse; too small and
floating-point cancellation becomes visible.

## Analytical vs Numerical Gradients

TensorTrail compares two gradient sources:

- analytical gradients from the dynamic graph and `backward()`
- numerical gradients from finite differences

The checker returns:

- `passed`
- `max_error`
- `max_rel_error`
- analytical gradients
- numerical gradients
- failure messages for mismatched entries

Example:

```python
from tensortrail import Tensor, gradcheck

x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
result = gradcheck(lambda value: (value * value).sum(), x)

print(result.passed)
print(result.max_error)
```

## Limitations

Gradient checking is slow because it evaluates the function twice for every
input element. It is meant for small test cases, not training loops.

It also works best for smooth functions. At non-smooth points, such as ReLU at
zero or max pooling ties, a numerical gradient may disagree with the chosen
analytical subgradient. That disagreement can be expected and does not always
mean the implementation is wrong.

## How TensorTrail Uses It

TensorTrail uses gradient checking in tests for smooth expressions and core
operations. It is especially useful when adding new backward logic, such as
matrix multiplication, reductions, activations, and convolution.

The project still pairs gradient checking with targeted exact tests for cases
where finite differences are less reliable, such as max pooling gradient
routing.
