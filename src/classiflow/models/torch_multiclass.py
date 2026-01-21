"""Sklearn-compatible PyTorch multiclass estimators."""

from __future__ import annotations

from typing import Any

from classiflow.backends.torch.estimators import (
    TorchSoftmaxRegressionClassifier,
    TorchMLPMulticlassClassifier,
)


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
        num_workers: int = 0,
    ):
        self.random_state = random_state
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            dropout=0.0,
            hidden_dim=128,
            n_layers=1,
            patience=0,
            seed=random_state,
            device=device,
            torch_dtype="float32",
            num_workers=num_workers,
            class_weight=class_weight,
            val_fraction=0.0,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params: Any) -> "TorchLinearClassifier":
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
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
        device: str = "cpu",
        num_workers: int = 0,
    ):
        self.random_state = random_state
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            dropout=0.0,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            patience=0,
            seed=random_state,
            device=device,
            torch_dtype="float32",
            num_workers=num_workers,
            class_weight=class_weight,
            val_fraction=0.0,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params: Any) -> "TorchMLPClassifier":
        if "random_state" in params:
            self.random_state = params["random_state"]
            self.seed = params["random_state"]
        return super().set_params(**params)
