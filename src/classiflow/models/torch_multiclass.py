"""Sklearn-compatible PyTorch multiclass estimators."""

from __future__ import annotations

from typing import Any

from classiflow.backends.torch.estimators import (
    TorchMLPMulticlassClassifier,
    TorchSoftmaxRegressionClassifier,
)
from classiflow.config import default_torch_num_workers


class TorchLinearClassifier(TorchSoftmaxRegressionClassifier):
    """Multiclass softmax regression with sklearn-compatible interface."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 25,
        batch_size: int = 256,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
        device: str = "cpu",
        num_workers: int | None = None,
        gpu_index_batching: bool = True,
        temperature_scaling: bool = False,
    ):
        resolved_workers = default_torch_num_workers() if num_workers is None else num_workers
        self.random_state = random_state
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            dropout=0.0,
            hidden_dim=128,
            n_layers=1,
            activation="relu",
            use_batchnorm=False,
            patience=0,
            seed=random_state,
            device=device,
            torch_dtype="float32",
            num_workers=resolved_workers,
            class_weight=class_weight,
            val_fraction=0.0,
            gpu_index_batching=gpu_index_batching,
            temperature_scaling=temperature_scaling,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params: Any) -> TorchLinearClassifier:
        if "random_state" in params:
            self.random_state = params["random_state"]
            self.seed = params["random_state"]
        return super().set_params(**params)


class TorchMLPClassifier(TorchMLPMulticlassClassifier):
    """Multiclass MLP classifier with 1-2 hidden layers."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        use_batchnorm: bool = False,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
        device: str = "cpu",
        num_workers: int | None = None,
        gpu_index_batching: bool = True,
        temperature_scaling: bool = False,
    ):
        resolved_workers = default_torch_num_workers() if num_workers is None else num_workers
        self.random_state = random_state
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_batchnorm=use_batchnorm,
            patience=0,
            seed=random_state,
            device=device,
            torch_dtype="float32",
            num_workers=resolved_workers,
            class_weight=class_weight,
            val_fraction=0.0,
            gpu_index_batching=gpu_index_batching,
            temperature_scaling=temperature_scaling,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params: Any) -> TorchMLPClassifier:
        if "random_state" in params:
            self.random_state = params["random_state"]
            self.seed = params["random_state"]
        return super().set_params(**params)
