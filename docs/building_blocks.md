# Building Blocks

TensorTrail is built from small pieces that compose into trainable models. This
page explains the user-facing building blocks and how the examples use them.

## Layers

All trainable layers inherit from `Module`. A module can expose parameters,
clear gradients, switch train/eval mode, and return non-trainable buffers for
serialization.

Common layers:

- `Linear`: fully connected affine transform.
- `Conv2D`: educational NCHW convolution with manual backward logic.
- `MaxPool2D`: routes gradients to max locations.
- `AveragePool2D`: distributes gradients evenly over pooling windows.
- `Flatten`: reshapes each batch item into a feature vector.
- `BatchNorm1D`: normalizes batch features and tracks running statistics.
- `LayerNorm`: normalizes each sample over the final dimension.
- `Dropout`: inverted dropout during training, identity during eval.
- `ReLU`, `Sigmoid`, `Tanh`: activation modules.
- `Sequential`: applies layers in order.

Example:

```python
from tensortrail import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential

model = Sequential(
    Conv2D(1, 4, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(2),
    Flatten(),
    Linear(4 * 4 * 4, 3),
)
```

## Losses

Loss functions turn model predictions into scalar tensors that can call
`backward()`.

- `MSELoss`: regression-style squared error.
- `BinaryCrossEntropyLoss`: binary classification from probabilities.
- `CrossEntropyLoss`: multi-class classification from raw logits.

Because losses are implemented with TensorTrail operations, their gradients
flow through the same dynamic graph as every other tensor operation.

## Optimizers

Optimizers update trainable tensors in place using accumulated gradients.

- `SGD`: simple gradient descent.
- `SGD(..., momentum=...)`: momentum variant.
- `Adam`: adaptive first/second moment optimizer.
- `StepLR`: decay the learning rate every fixed number of epochs.
- `ExponentialLR`: decay the learning rate every epoch.

All optimizers also expose `clip_grad_norm(max_norm)`, which rescales current
gradients by global L2 norm before an update. This is useful when an example's
loss surface produces very large gradients.

Typical pattern:

```python
loss = loss_fn(model(x_batch), y_batch)
optimizer.zero_grad()
loss.backward()
optimizer.clip_grad_norm(1.0)
optimizer.step()
```

Schedulers wrap an optimizer and update its `lr` over time:

```python
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
```

## DataLoader

`Dataset` stores feature and target arrays. `DataLoader` creates mini-batches
and yields TensorTrail tensors:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, seed=0)
for x_batch, y_batch in loader:
    ...
```

The project uses offline synthetic datasets so examples run without downloads.

## Trainer

`Trainer` packages the standard supervised training loop:

- iterate over batches
- compute predictions and loss
- run backward
- update parameters
- evaluate validation data
- log metrics
- optionally clip gradients before optimizer steps
- optionally step a learning-rate scheduler once per epoch
- optionally early-stop and save checkpoints

Example:

```python
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    optimizer=optimizer,
    metrics=["accuracy"],
    clip_grad_norm=1.0,
    scheduler=scheduler,
)

history = trainer.fit(train_loader, val_loader=val_loader, epochs=20)
results = trainer.evaluate(val_loader)
```

## How The Examples Compose The Blocks

- `train_xor.py`: tensors, MLP layers, binary cross entropy, Adam.
- `train_mnist_like.py`: synthetic data, multi-class MLP, cross entropy.
- `train_mlp_classifier.py`: DataLoader, BatchNorm, Dropout, Trainer.
- `train_regularized_mlp.py`: deeper regularized MLP with validation metrics.
- `train_tiny_cnn.py`: Conv2D, pooling, Flatten, Linear, Trainer.
- `visualize_autograd_graph.py`: graph construction and DOT export.
- `benchmark_ops.py`: educational comparison between TensorTrail and NumPy.
