
from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def function_to_module(torch_function: Callable) -> nn.Module:
    """Converts a torch function to a torch module."""
    class ModularizedFunction(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch_function(x)

    return ModularizedFunction


ACTIVATION_NAME_MAP = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "softplus": torch.nn.Softplus,
    "silu": torch.nn.SiLU,
    "identity": torch.nn.Identity,
    'abs': function_to_module(torch.abs),
}

LOSS_NAME_MAP = {
    "ce": nn.CrossEntropyLoss,
    "bcewl": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}

METRIC_TO_FUNCTION = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "matthews_correlation": matthews_corrcoef,
}

