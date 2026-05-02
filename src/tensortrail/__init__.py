"""TensorTrail: forging tensors, gradients, and neural networks from scratch."""

from .data import DataLoader, Dataset, make_mnist_like, make_xor, train_test_split
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .modules import Linear, Module, ReLU, Sequential, Sigmoid, Tanh
from .optim import Adam, SGD
from .tensor import Tensor, tensor
from .trainer import Trainer

__all__ = [
    "Adam",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "DataLoader",
    "Dataset",
    "Linear",
    "MSELoss",
    "Module",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "Tensor",
    "Trainer",
    "make_mnist_like",
    "make_xor",
    "tensor",
    "train_test_split",
]

