"""Torch modules for binary and multiclass classifiers."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BinaryLinear(nn.Module):
    """Single-layer linear model for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class BinaryMLP(nn.Module):
    """MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MulticlassLinear(nn.Module):
    """Linear softmax regression for multiclass classification."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MulticlassMLP(nn.Module):
    """MLP for multiclass classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
