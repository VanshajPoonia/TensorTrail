# TensorTrail

TensorTrail is a handcrafted neural network framework built from scratch in Python with NumPy: forging tensors, gradients, and neural networks from scratch.

It is intentionally small, readable, and educational. The goal is not to compete with PyTorch or TensorFlow, but to show how the core ideas behind modern ML frameworks fit together: tensors, computational graphs, reverse-mode automatic differentiation, modules, losses, optimizers, data loading, and training loops.
## Why This Exists

Most machine learning projects use mature frameworks, which is the right choice for production. TensorTrail takes the opposite route for learning value: every important piece is implemented directly so the mechanics are visible.

This makes TensorTrail a resume-quality systems-and-ML project. It demonstrates numerical programming, API design, graph-based differentiation, neural network training, testing discipline, and documentation.
## Features

- NumPy-backed `Tensor` object
- Reverse-mode autograd engine with topological graph traversal
- Broadcasting-aware backward passes
- Differentiable arithmetic, reductions, matrix multiplication, reshaping, transpose, and common activations
- Stable softmax and log-softmax helpers
- Minimal module system with `Linear`, `ReLU`, `Sigmoid`, `Tanh`, and `Sequential`
- `MSELoss`, `BinaryCrossEntropyLoss`, and `CrossEntropyLoss`
- `SGD`, SGD with momentum, and `Adam`
- Offline `Dataset`, `DataLoader`, train/test split, XOR data, and synthetic MNIST-like data
- Simple `Trainer` utility
- Runnable examples and pytest coverage
## Installation

From the repository root:

```bash
cd tensortrail
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Or install the lightweight requirements file:

```bash
pip install -r requirements.txt
```
## Quick Usage

```python
from tensortrail import BinaryCrossEntropyLoss, Linear, Sequential, Sigmoid, Tanh, Tensor
from tensortrail.optim import Adam

x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

model = Sequential(
    Linear(2, 8),
    Tanh(),
    Linear(8, 1),
    Sigmoid(),
)

loss_fn = BinaryCrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.05)

for _ in range(1000):
    predictions = model(x)
    loss = loss_fn(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(x).data)
```
## Running Examples

```bash
python examples/train_xor.py
python examples/train_mnist_like.py
```

`train_xor.py` trains a tiny MLP to learn the XOR truth table. `train_mnist_like.py` creates a synthetic flattened-image classification dataset and trains a small classifier without downloading anything.
## Architecture

TensorTrail is organized around a few small building blocks:

- `tensor.py` defines `Tensor`, graph tracking, and operation-level gradient rules.
- `ops.py` exposes reusable operations such as `softmax`, `log_softmax`, `one_hot`, and `accuracy`.
- `modules.py` provides neural network layers and composition.
- `losses.py` defines objective functions.
- `optim.py` updates trainable parameters in place.
- `data.py` and `trainer.py` provide enough infrastructure to run complete training experiments.
## How Autograd Works

Each differentiable operation creates a new `Tensor` containing:

- computed NumPy data
- references to parent tensors
- a small backward closure describing how output gradients flow to parents

Calling `backward()` topologically sorts the dynamic computation graph, seeds the output gradient, and runs the backward closures in reverse order. Gradients accumulate into each tensor's `.grad`, so reused tensors correctly receive contributions from multiple branches.

Broadcasting is handled by reducing output gradients back to each operand's original shape before accumulation.
