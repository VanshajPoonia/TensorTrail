"""TensorTrail: forging tensors, gradients, and neural networks from scratch."""

from .data import DataLoader, Dataset, make_mnist_like, make_xor, train_test_split
from .gradcheck import GradCheckResult, gradcheck
from .graph import visualize_graph
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .modules import (
    BatchNorm1D,
    Dropout,
    Flatten,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from .optim import Adam, SGD
from .serialization import load_model, save_model
from .tensor import Tensor, tensor
from .trainer import Trainer

__all__ = [
    "Adam",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "DataLoader",
    "Dataset",
    "BatchNorm1D",
    "Dropout",
    "Flatten",
    "GradCheckResult",
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
    "gradcheck",
    "load_model",
    "make_mnist_like",
    "make_xor",
    "save_model",
    "tensor",
    "train_test_split",
    "visualize_graph",
]
