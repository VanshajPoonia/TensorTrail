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
