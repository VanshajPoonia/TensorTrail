"""TensorTrail: forging tensors, gradients, and neural networks from scratch."""

from .data import DataLoader, Dataset, make_mnist_like, make_xor, train_test_split
from .gradcheck import GradCheckResult, gradcheck
from .graph import visualize_graph
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .metrics import accuracy_score, binary_accuracy
from .modules import (
    AveragePool2D,
    BatchNorm1D,
    Conv2D,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2D,
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
    "AveragePool2D",
    "BatchNorm1D",
    "BinaryCrossEntropyLoss",
    "Conv2D",
    "CrossEntropyLoss",
    "DataLoader",
    "Dataset",
    "Dropout",
    "Flatten",
    "GradCheckResult",
    "Linear",
    "MaxPool2D",
    "MSELoss",
    "Module",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "Tensor",
    "Trainer",
    "LayerNorm",
    "accuracy_score",
    "binary_accuracy",
    "gradcheck",
    "load_model",
    "make_mnist_like",
    "make_xor",
    "save_model",
    "tensor",
    "train_test_split",
    "visualize_graph",
]
