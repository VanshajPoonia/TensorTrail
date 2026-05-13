# Framework Architecture

TensorTrail is organized as a compact deep learning stack. Each file owns one
clear idea so the project stays readable and easy to extend.

## `tensor.py`

Defines the `Tensor` object and reverse-mode autograd engine. Tensor operations
store output data, parent references, operation names, and backward closures.
Calling `backward()` topologically sorts the dynamic graph and accumulates
gradients into tensors that require them.

Core responsibilities:

- numeric storage through NumPy arrays
- scalar and external-gradient backward support
- broadcasting-aware gradient reduction
- arithmetic, matrix multiplication, reductions, reshape, transpose, and activations
- clear error messages for invalid shapes and unsupported dtypes

## `ops.py`

Contains reusable functions that sit above the core `Tensor` class:

- numerically stable `softmax` and `log_softmax`
- `one_hot`
- classification `accuracy`

These operations are expressed with TensorTrail tensors where gradients matter.
Elementwise activations live as `Tensor` methods and module wrappers in
`modules.py`.

## `modules.py`

Defines the neural network layer system. `Module` handles:

- discovering trainable `Tensor` parameters
- clearing gradients
- exposing checkpoint buffers
- switching train/eval mode recursively

Implemented layers include:

- dense layers: `Linear`
- convolution and pooling: `Conv2D`, `MaxPool2D`, `AveragePool2D`
- shape transforms: `Flatten`
- normalization: `BatchNorm1D`, `LayerNorm`
- regularization: `Dropout`
- activations: `ReLU`, `Sigmoid`, `Tanh`
- composition: `Sequential`

CNN layers use explicit NumPy loops with custom backward closures. This keeps
the implementation readable and testable, but not optimized for large image
workloads.

## `losses.py`

Implements objective functions using TensorTrail operations so they remain
differentiable:

- `MSELoss`
- `BinaryCrossEntropyLoss`
- `CrossEntropyLoss`

`CrossEntropyLoss` uses `log_softmax` and one-hot targets for stable
multi-class classification.

## `optim.py`

Updates parameters in place:

- `SGD`
- SGD with momentum
- `Adam`
- global gradient clipping by L2 norm
- epoch-level `StepLR` and `ExponentialLR` schedules

Optimizers operate directly on `Tensor.data` and `Tensor.grad`, keeping the
training loop simple. Schedulers mutate the optimizer's `lr` without changing
the optimizer's accumulated state.

## `data.py`

Provides offline data utilities:

- `Dataset`
- `DataLoader`
- `train_test_split`
- `make_xor`
- `make_mnist_like`

The project avoids external dataset downloads so examples run anywhere.

## `trainer.py`

Holds the supervised training loop. `Trainer` supports:

- Dataset or DataLoader inputs
- validation loaders
- named metrics such as `"accuracy"`
- logging intervals
- optional gradient clipping before optimizer steps
- optional learning-rate scheduler step once per epoch
- early stopping
- optional best-checkpoint saving
- `evaluate()` for validation or test sets

The trainer is intentionally small enough to read while still showing how a real
framework connects models, losses, optimizers, metrics, and data.

## `metrics.py`

Provides NumPy-based evaluation helpers:

- `accuracy_score`
- `binary_accuracy`

These functions accept TensorTrail tensors or NumPy-like arrays.

## `gradcheck.py`

Implements centered finite-difference gradient checking. It compares numerical
gradients against TensorTrail's analytical gradients from `backward()`, then
returns a result object with pass/fail status, max error, and both gradient
sets.

## `serialization.py`

Saves and loads model state with NumPy `.npz` files. Trainable parameters are
stored in order as `param_N`; non-trainable buffers such as BatchNorm running
statistics are stored as `buffer_N`.

Loading validates parameter count and shape to catch mismatched model
structures. Named checkpoints are also supported with module paths such as
`param:layers.0.weight` for clearer state inspection.

## `graph.py`

Exports a TensorTrail computation graph as Graphviz DOT. The file is static and
dependency-free: each node shows tensor shape, operation name, and whether it
requires gradients.

Graphviz is optional. TensorTrail only writes the DOT text file.

## `examples/`

The examples are part of the architecture story. They demonstrate:

- scalar and binary classification with XOR
- multi-class MLP training
- regularization with BatchNorm and Dropout
- tiny CNN training on synthetic image-like data
- graph export
- educational operation benchmarking
