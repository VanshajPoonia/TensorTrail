# Framework Architecture

TensorTrail is organized as a compact deep learning stack. Each module owns one
clear idea so the project stays readable.

## `tensor.py`

Defines the `Tensor` object and reverse-mode autograd engine. Tensor operations
store output data, parent references, operation names, and backward closures.
Calling `backward()` topologically sorts the graph and accumulates gradients.

## `ops.py`

Provides reusable tensor operations that are easier to express outside the core
class, including softmax, log-softmax, one-hot encoding, and accuracy.

## `modules.py`

Contains the neural network layer abstraction. `Module` discovers trainable
parameters, switches train/eval mode, and exposes checkpoint buffers. Layers
include `Linear`, activations, `Sequential`, `Flatten`, `Dropout`, and
`BatchNorm1D`.

## `losses.py`

Implements objective functions using TensorTrail operations, so losses remain
differentiable. Current losses include mean squared error, binary cross entropy,
and multi-class cross entropy.

## `optim.py`

Updates trainable tensors in place. `SGD` supports optional momentum, and `Adam`
tracks first and second moments for adaptive updates.

## `data.py`

Provides tiny offline data utilities: an in-memory `Dataset`, mini-batch
`DataLoader`, train/test splitting, XOR data, and synthetic MNIST-like data.

## `trainer.py`

Holds the supervised training loop. It handles mini-batches, validation,
metrics, progress logs, early stopping, and optional best-checkpoint saving.

## `serialization.py`

Saves and loads model state with NumPy `.npz` files. Trainable parameters are
stored as `param_N`; non-trainable buffers such as BatchNorm running statistics
are stored as `buffer_N`.

## `graph.py`

Exports a TensorTrail computation graph as Graphviz DOT. The export is static
and dependency-free: each node shows tensor shape, operation name, and whether
it requires gradients.

