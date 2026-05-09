# TensorTrail

**A from-scratch neural network framework forged with NumPy.**

TensorTrail is a compact educational deep learning framework built directly on
NumPy. It includes a small reverse-mode autograd engine, neural network layers,
losses, optimizers, data utilities, a trainer API, model serialization,
gradient checking, graph export, and runnable examples.

It is not trying to replace PyTorch or TensorFlow. The point is to make the
core machinery of a neural network framework visible, readable, and testable.

## Why I Built This

I built TensorTrail to understand deep learning infrastructure from the inside:
how tensors remember computation history, how gradients flow backward through a
dynamic graph, how layers expose parameters, and how training loops connect
data, models, losses, optimizers, metrics, and checkpoints.

The project is intentionally small enough to inspect, but complete enough to
train MLPs and tiny CNNs on offline synthetic datasets.

## Feature Checklist

- NumPy-backed `Tensor` object
- Reverse-mode automatic differentiation
- Dynamic computation graph with operation names
- Topological backward pass
- Broadcasting-aware gradients
- Gradient checker with centered finite differences
- Graphviz DOT computation graph export
- Layers: `Linear`, `Conv2D`, pooling, normalization, dropout, activations
- `Sequential` model composition
- Losses: mean squared error, binary cross entropy, multi-class cross entropy
- Optimizers: SGD, momentum SGD, Adam
- Offline datasets and mini-batch `DataLoader`
- Trainer with validation, metrics, early stopping, and checkpoint saving
- NumPy `.npz` model save/load
- Runnable examples and pytest coverage

## Installation

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

If your shell exposes Python as `python3`, use:

```bash
python3 -m pip install -e .
```

## Quickstart

```python
from tensortrail import Linear, ReLU, Sequential, Tensor

x = Tensor([[1.0, 2.0, 3.0]])
model = Sequential(
    Linear(3, 8, seed=1),
    ReLU(),
    Linear(8, 2, seed=2),
)

logits = model(x)
print(logits)
```

## XOR Example

```bash
python examples/train_xor.py
```

This trains a tiny MLP on the XOR truth table:

```python
model = Sequential(
    Linear(2, 8),
    Tanh(),
    Linear(8, 1),
    Sigmoid(),
)
```

## Trainer API

```python
from tensortrail import (
    CrossEntropyLoss, DataLoader, Linear, ReLU, Sequential, Trainer,
)
from tensortrail.data import train_test_split, make_mnist_like
from tensortrail.optim import Adam

dataset = make_mnist_like(n_samples=800, n_features=32, n_classes=4, seed=0)
train_data, val_data = train_test_split(dataset.x, dataset.y, test_size=0.2, seed=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, seed=42)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

model = Sequential(Linear(32, 16, seed=1), ReLU(), Linear(16, 4, seed=2))

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    optimizer=Adam(model.parameters(), lr=0.01),
    metrics=["accuracy"],          # built-in string shorthand
)

history = trainer.fit(
    train_loader,
    val_loader=val_loader,
    epochs=20,
    log_every=5,                   # print every 5 epochs
    early_stopping_patience=5,     # stop if val_loss stalls
    checkpoint_path="best.npz",    # save best weights
)

# History keys: epoch, train_loss, val_loss, accuracy, val_accuracy, elapsed_time
print(history["val_accuracy"])

# Evaluate on a held-out set
results = trainer.evaluate(val_loader)
print(f"val loss={results['loss']:.4f}  accuracy={results['accuracy']:.4f}")
```

`trainer.fit()` returns a history dictionary with one list per key and one
entry per completed epoch.  When a `val_loader` is supplied the dictionary also
includes `val_loss` and `val_<metric>` keys.

`trainer.evaluate()` runs a single pass in eval mode and returns a plain
`dict[str, float]` with `"loss"` and each configured metric.

## MLP Classifier Example

```bash
python examples/train_mlp_classifier.py
```

The classifier example generates an offline synthetic multi-class dataset and
trains:

```python
model = Sequential(
    Linear(32, 24),
    BatchNorm1D(24),
    ReLU(),
    Dropout(p=0.15),
    Linear(24, 4),
)
```

It uses `DataLoader`, `metrics=["accuracy"]`, and `trainer.evaluate()`, then
prints train loss, validation loss, and test accuracy so the learning curve is
visible from the terminal.

## Tiny CNN Example

```bash
python examples/train_tiny_cnn.py
```

The tiny CNN example generates offline `8x8` grayscale images with simple
class-specific patterns, then trains:

```python
model = Sequential(
    Conv2D(1, 4, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    Flatten(),
    Linear(4 * 4 * 4, 3),
)
```

`Conv2D`, `MaxPool2D`, and `AveragePool2D` are educational, loop-based
implementations. They are designed for correctness and readability on small
inputs, not high-performance computer vision workloads.

## Computation Graph Visualization

```bash
python examples/visualize_autograd_graph.py
```

This writes `graph.dot`, a dependency-free Graphviz DOT file. Each node shows:

- tensor shape
- operation name
- whether gradients are tracked

You can render it later with Graphviz:

```bash
dot -Tpng graph.dot -o graph.png
```

Graphviz is optional; TensorTrail only writes the DOT text file.

## Supported Layers

| Layer | Class | Notes |
|---|---|---|
| Fully connected | `Linear(in, out)` | Xavier uniform init, optional bias |
| 2D convolution | `Conv2D(in_channels, out_channels, kernel_size)` | NCHW input, stride/padding, educational loop implementation |
| Max pooling | `MaxPool2D(kernel_size)` | NCHW input, routes gradients to max positions |
| Average pooling | `AveragePool2D(kernel_size)` | NCHW input, distributes gradients evenly |
| Flatten | `Flatten()` | `(batch, …) → (batch, features)` |
| Batch normalisation | `BatchNorm1D(features)` | learnable γ/β, running stats, train/eval modes |
| Layer normalisation | `LayerNorm(features)` | per-sample normalisation over last dim, learnable γ/β |
| Dropout | `Dropout(p)` | inverted dropout, inactive in eval mode |
| ReLU | `ReLU()` | |
| Sigmoid | `Sigmoid()` | |
| Tanh | `Tanh()` | |
| Container | `Sequential(*layers)` | chains modules in order |

All layers inherit from `Module` and support:
- `parameters()` — returns learnable `Tensor` objects
- `buffers()` — returns non-trainable state (e.g. BatchNorm running stats)
- `train()` / `eval()` — switch training mode recursively

## Architecture Overview

- `tensor.py` implements the Tensor object and autograd engine.
- `ops.py` contains reusable functions such as softmax and accuracy.
- `modules.py` defines layers, `Sequential`, train/eval mode, parameters, and buffers.
- `losses.py` implements differentiable objective functions.
- `optim.py` updates parameters with SGD or Adam.
- `data.py` provides small offline datasets and mini-batches.
- `trainer.py` runs training, validation, metrics, early stopping, and checkpoints.
- `serialization.py` saves and loads NumPy `.npz` model state.
- `gradcheck.py` compares autograd gradients against finite differences.
- `graph.py` exports autograd graphs as Graphviz DOT.

For deeper explanations, see:

- `docs/autograd_explained.md`
- `docs/framework_architecture.md`

## Core Stability

TensorTrail's core tests focus on the pieces that make an autodiff framework
trustworthy:

- scalar and non-scalar `backward()` behavior
- gradient accumulation when a tensor feeds multiple graph branches
- broadcasting-aware gradients for bias-like tensors
- arithmetic, division, power, matrix multiplication, reductions, reshape, and transpose gradients
- activation gradients for ReLU, sigmoid, and tanh
- helpful errors for unsupported dtypes, missing external gradients, and invalid matmul shapes

This keeps new layers and examples grounded in a small autograd engine whose
behavior is directly tested.

## What I Built From Scratch

- Tensor object
- Reverse-mode autodiff
- Computational graph
- Topological backward pass
- Broadcasting-aware gradient handling
- Layer abstraction
- Loss functions
- Optimizers
- Training loop
- Validation and metrics
- Gradient checking
- Model serialization
- Graph visualization
- Example models

TensorTrail does not use PyTorch, TensorFlow, JAX, autograd, tinygrad,
micrograd, or any existing ML/autograd framework. NumPy is used as the numerical
array backend.

## Testing

```bash
pytest
```

Equivalent command if `pytest` is not installed as a standalone executable:

```bash
python3 -m pytest
```

The tests cover tensor operations, autograd, broadcasting, modules, losses,
optimizers, serialization, gradient checking, graph export, and trainer behavior.

## Validation Commands

These are the main smoke checks for the project:

```bash
python examples/train_xor.py
python examples/train_mnist_like.py
python examples/train_mlp_classifier.py
python examples/train_tiny_cnn.py
python examples/visualize_autograd_graph.py
pytest
```

On systems where only `python3` is available, replace `python` with `python3`.

## Roadmap

- Add gradient clipping
- Add learning-rate schedules
- Add richer plotting for training histories
- Add notebook walkthroughs
- Add a tiny experiment registry for examples

## Limitations

TensorTrail is educational infrastructure, not production ML infrastructure. It
does not target GPUs, distributed training, mixed precision, automatic batching,
large datasets, deployment, or high performance.

That tradeoff is deliberate. The project favors clarity and inspectability so
the trail from tensors to trained neural networks stays visible.
