# TensorTrail Examples

All examples run offline. They generate synthetic data locally and do not
download external datasets.

Run them from the repository root after installing TensorTrail:

```bash
pip install -e .
```

## `train_xor.py`

Trains a tiny MLP on the XOR truth table with binary cross entropy and Adam.

```bash
python examples/train_xor.py
```

Use this as the fastest sanity check that tensors, layers, losses, optimizers,
and backpropagation are connected correctly.

## `train_binary_classifier.py`

Trains a small binary classifier from raw logits with `BCEWithLogitsLoss`.

```bash
python examples/train_binary_classifier.py
```

Use this when you want a binary example that avoids putting `Sigmoid` in the
model itself.

## `train_mnist_like.py`

Trains an MLP on a synthetic flattened-image classification dataset.

```bash
python examples/train_mnist_like.py
```

This keeps the spirit of an MNIST-style task while avoiding downloads.

## `train_mlp_classifier.py`

Trains a multi-class MLP with `DataLoader`, `BatchNorm1D`, `Dropout`,
validation metrics, early stopping, and `trainer.evaluate()`.

```bash
python examples/train_mlp_classifier.py
```

This is the best example for the higher-level Trainer API.

## `train_regularized_mlp.py`

Shows a deeper regularized MLP with BatchNorm and Dropout in train/eval mode.

```bash
python examples/train_regularized_mlp.py
```

Use this to inspect how regularization layers behave inside `Sequential`.

## `train_tiny_cnn.py`

Trains a tiny CNN on synthetic `8x8` grayscale images with class-specific
patterns.

```bash
python examples/train_tiny_cnn.py
```

This demonstrates `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Linear`, and the
Trainer API. The convolution and pooling layers are educational loop-based
implementations, so the example intentionally stays small.

## `visualize_autograd_graph.py`

Exports a small computation graph to Graphviz DOT.

```bash
python examples/visualize_autograd_graph.py
```

The script writes `graph.dot`. Graphviz is optional; TensorTrail only generates
the text DOT file.

## `benchmark_ops.py`

Runs a small educational benchmark comparing TensorTrail operations with raw
NumPy.

```bash
python examples/benchmark_ops.py
```

This is not a performance claim. It is meant to show the overhead of a readable
Python autograd framework compared with direct NumPy calls.

## `plot_training_history.py`

Plots `Trainer.fit` loss and accuracy history for an offline synthetic
classification dataset.

```bash
python -m pip install -e ".[plot]"
python examples/plot_training_history.py
```

This optional example writes `training_history.png` and is kept out of the
default smoke command because Matplotlib is not a runtime dependency.
