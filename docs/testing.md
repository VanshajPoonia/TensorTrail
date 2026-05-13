# Testing TensorTrail

TensorTrail's tests are meant to make the framework trustworthy without hiding
how it works. The suite favors small, direct examples over large fixtures.

## What The Tests Cover

- Tensor creation, dtype validation, scalar and non-scalar backward calls.
- Reverse-mode autograd for arithmetic, broadcasting, matmul, reductions,
  reshaping, transposes, exponentials, logarithms, and activations.
- Layer behavior for dense layers, normalization, dropout, CNN layers, pooling,
  flattening, and sequential composition.
- Losses, optimizers, data utilities, trainer workflows, metrics, model
  serialization, gradient checking, and graph export.
- Gradient clipping through optimizers and the high-level trainer.
- Example-level workflows through fast synthetic datasets.

## Commands

```bash
python -m pytest
```

If your environment exposes Python as `python3`:

```bash
python3 -m pytest
```

The full portfolio smoke check is:

```bash
python examples/train_xor.py
python examples/train_mnist_like.py
python examples/train_mlp_classifier.py
python examples/train_regularized_mlp.py
python examples/train_tiny_cnn.py
python examples/visualize_autograd_graph.py
python examples/benchmark_ops.py
python -m pytest
```

All examples use offline synthetic data and should complete quickly on CPU.

The optional plotting example requires Matplotlib:

```bash
python -m pip install -e ".[plot]"
python examples/plot_training_history.py
```
