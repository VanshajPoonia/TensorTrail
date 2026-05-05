# TensorTrail

TensorTrail: forging tensors, gradients, and neural networks from scratch.

TensorTrail is a compact deep learning framework built from scratch with Python
and NumPy. It is intentionally educational: the code is small enough to read,
but complete enough to train neural networks, inspect autograd graphs, checkpoint
models, and test gradients.

The goal is not to beat PyTorch or TensorFlow. The goal is to show that the core
pieces of a modern neural network framework can be built directly: tensors,
reverse-mode autodiff, layers, losses, optimizers, training loops, model
serialization, graph visualization, and tests.

## Feature Checklist

- NumPy-backed `Tensor` object
- Reverse-mode automatic differentiation
- Dynamic computation graph with operation names
- Broadcasting-aware gradients
- Gradient checker with finite differences
- Graphviz DOT computation graph export
- Layers: `Linear`, `Flatten`, `Dropout`, `BatchNorm1D`, `ReLU`, `Sigmoid`, `Tanh`
- `Sequential` model composition
- Losses: MSE, binary cross entropy, multi-class cross entropy
- Optimizers: SGD, momentum SGD, Adam
- Offline datasets and mini-batch `DataLoader`
- Trainer with validation, metrics, early stopping, and checkpoint saving
- NumPy `.npz` model save/load
- Runnable examples and pytest coverage

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
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

It prints train loss, validation loss, and validation accuracy so the learning
curve is visible from the terminal.

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

The tests cover tensor operations, autograd, broadcasting, modules, losses,
optimizers, serialization, gradient checking, graph export, and trainer behavior.

## Roadmap

- Add convolution and pooling layers
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

