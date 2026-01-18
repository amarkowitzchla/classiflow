"""Torch backend estimators and utilities."""

from classiflow.backends.torch.estimators import (
    TorchLogisticRegressionClassifier,
    TorchMLPClassifier,
    TorchSoftmaxRegressionClassifier,
    TorchMLPMulticlassClassifier,
)
from classiflow.backends.torch.utils import resolve_device

__all__ = [
    "TorchLogisticRegressionClassifier",
    "TorchMLPClassifier",
    "TorchSoftmaxRegressionClassifier",
    "TorchMLPMulticlassClassifier",
    "resolve_device",
]
