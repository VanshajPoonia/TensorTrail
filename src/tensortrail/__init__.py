"""TensorTrail: forging tensors, gradients, and neural networks from scratch."""

from .data import DataLoader, Dataset, make_mnist_like, make_xor, train_test_split
from .gradcheck import GradCheckResult, gradcheck
from .graph import visualize_graph
from .losses import BCEWithLogitsLoss, BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
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
from .optim import Adam, ExponentialLR, LRScheduler, SGD, StepLR
from .ops import accuracy, log_softmax, one_hot, softmax
from .serialization import load_model, save_model
from .tensor import Tensor, tensor
from .trainer import Trainer, classification_accuracy

__all__ = [
    "Adam",
    "AveragePool2D",
    "BCEWithLogitsLoss",
    "BatchNorm1D",
    "BinaryCrossEntropyLoss",
    "Conv2D",
    "CrossEntropyLoss",
    "DataLoader",
    "Dataset",
    "Dropout",
    "ExponentialLR",
    "Flatten",
    "GradCheckResult",
    "LayerNorm",
    "Linear",
    "LRScheduler",
    "MaxPool2D",
    "MSELoss",
    "Module",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "StepLR",
    "Tanh",
    "Tensor",
    "Trainer",
    "accuracy",
    "accuracy_score",
    "binary_accuracy",
    "classification_accuracy",
    "gradcheck",
    "load_model",
    "log_softmax",
    "make_mnist_like",
    "make_xor",
    "one_hot",
    "save_model",
    "softmax",
    "tensor",
    "train_test_split",
    "visualize_graph",
]
