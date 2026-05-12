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
python3 -m pip install -e ".[dev]"
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

## MLP Classifier Example

```bash
python examples/train_mlp_classifier.py
```

This example generates an offline synthetic multi-class dataset and trains an
MLP with BatchNorm and Dropout:

```python
model = Sequential(
    Linear(32, 24),
    BatchNorm1D(24),
    ReLU(),
    Dropout(p=0.15),
    Linear(24, 4),
)
```

It uses `DataLoader`, `Trainer(metrics=["accuracy"])`, validation, early
stopping, and `trainer.evaluate()`.

## CNN Example

```bash
python examples/train_tiny_cnn.py
```

The tiny CNN example generates offline `8x8` grayscale images with simple
class-specific patterns:

```python
model = Sequential(
    Conv2D(1, 4, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    Flatten(),
    Linear(4 * 4 * 4, 3),
)
```

`Conv2D`, `MaxPool2D`, and `AveragePool2D` are educational loop-based
implementations. They are correct and readable for small CPU examples, but not
optimized for production computer vision workloads.

## Autograd Explanation

TensorTrail records a dynamic computation graph as tensor operations run. Each
operation creates an output `Tensor` with references to its parents, an
operation name, and a small backward closure that knows the local derivative.

Calling `backward()` on a scalar output:

1. Builds a topological ordering from the output back to all parents.
2. Seeds the output gradient with `1`.
3. Walks the graph in reverse topological order.
4. Calls each node's backward closure.
5. Accumulates gradient contributions into every tensor that requires them.

Broadcasting is handled by reducing gradients back to each operand's original
shape, which is essential for bias terms and normalization parameters.

For a deeper walkthrough, see [docs/autograd_explained.md](docs/autograd_explained.md).

## Architecture Overview

- `tensor.py` implements the `Tensor` object and autograd engine.
- `ops.py` contains softmax, log-softmax, one-hot encoding, and accuracy.
- `modules.py` defines layers, `Sequential`, train/eval mode, parameters, and buffers.
- `losses.py` implements differentiable objective functions.
- `optim.py` updates trainable tensors with SGD or Adam.
- `data.py` provides offline datasets, splitting, and mini-batches.
- `trainer.py` runs training, validation, metrics, early stopping, and checkpoints.
- `gradcheck.py` compares analytical gradients with finite differences.
- `serialization.py` saves and loads NumPy `.npz` model state.
- `graph.py` exports computation graphs as Graphviz DOT.

More detail:

- [docs/framework_architecture.md](docs/framework_architecture.md)
- [docs/building_blocks.md](docs/building_blocks.md)
- [docs/gradient_checking.md](docs/gradient_checking.md)
- [docs/testing.md](docs/testing.md)

## Supported Layers

| Layer | Class | Notes |
|---|---|---|
| Fully connected | `Linear(in, out)` | Xavier uniform init, optional bias |
| 2D convolution | `Conv2D(in_channels, out_channels, kernel_size)` | NCHW input, stride, padding |
| Max pooling | `MaxPool2D(kernel_size)` | Routes gradients to max positions |
| Average pooling | `AveragePool2D(kernel_size)` | Distributes gradients evenly |
| Flatten | `Flatten()` | `(batch, ...) -> (batch, features)` |
| Batch normalization | `BatchNorm1D(features)` | Learnable scale/shift, running stats |
| Layer normalization | `LayerNorm(features)` | Per-sample normalization over last dimension |
| Dropout | `Dropout(p)` | Inverted dropout, inactive in eval mode |
| Activations | `ReLU`, `Sigmoid`, `Tanh` | Differentiable activation modules |
| Container | `Sequential(*layers)` | Chains modules in order |

## What I Built From Scratch

- Tensor object
- Reverse-mode automatic differentiation
- Computational graph construction
- Topological backward pass
- Broadcasting-aware gradients
- Neural network `Module` system
- Layers
- Loss functions
- Optimizers
- DataLoader
- Trainer API
- Gradient checker
- Model serialization
- Computation graph exporter
- Example training scripts

TensorTrail does not use PyTorch, TensorFlow, JAX, autograd, tinygrad,
micrograd, scikit-learn models, or any existing ML/autograd framework. NumPy is
the numerical backend.

## Testing

```bash
pytest
```

Equivalent command if `pytest` is not installed as a standalone executable:

```bash
python3 -m pytest
```

The tests cover tensor operations, autograd, broadcasting, modules, CNN layers,
losses, optimizers, serialization, gradient checking, graph export, data
utilities, metrics, and trainer behavior.

## Useful Commands

```bash
make install
make test
make xor
make examples
```

Main smoke checks:

```bash
python examples/train_xor.py
python examples/train_mnist_like.py
python examples/train_mlp_classifier.py
python examples/train_regularized_mlp.py
python examples/train_tiny_cnn.py
python examples/visualize_autograd_graph.py
python examples/benchmark_ops.py
pytest
```

See [examples/README.md](examples/README.md) for a guide to each script.

## Roadmap

- Add richer named checkpoint metadata on top of `.npz` files
- Add stricter validation and clearer error messages around user inputs
- Add optional Matplotlib plots for example training histories
- Add learning-rate schedules
- Add gradient clipping
- Add notebook walkthroughs
- Add an im2col Conv2D implementation for faster educational comparison

## Limitations

TensorTrail is educational infrastructure, not production ML infrastructure. It
does not target GPUs, distributed training, mixed precision, automatic
batching, large datasets, deployment, or high performance.

The Conv2D and pooling layers use explicit Python/NumPy loops for clarity. That
makes them easy to read and test, but much slower than optimized kernels in
production frameworks.

## Resume Bullets

- Built TensorTrail, a from-scratch neural network framework in Python and NumPy with reverse-mode automatic differentiation, dynamic computation graphs, and broadcasting-aware gradient accumulation.
- Implemented a topological backward pass, finite-difference gradient checker, and broadcasting-correct gradient reduction to validate autograd correctness across all operations.
- Developed a modular neural network API with `Tensor`, `Module`, `Linear`, `Conv2D`, pooling, `BatchNorm1D`, `LayerNorm`, `Dropout`, activation layers, `Sequential`, `MSELoss`, `CrossEntropyLoss`, `SGD`, `Adam`, `DataLoader`, and `Trainer`.
- Added NumPy `.npz` model serialization, Graphviz computation graph export, and a `Trainer` with validation, early stopping, and checkpoint saving.
- Validated the framework end-to-end with 111 pytest tests covering autograd, broadcasting, matmul, CNN layers, losses, optimizers, serialization, gradient checking, and trainer workflows.
