"""Torch estimator wrappers compatible with sklearn APIs."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from classiflow.backends.torch.modules import (
    BinaryLinear,
    BinaryMLP,
    MulticlassLinear,
    MulticlassMLP,
)
from classiflow.backends.torch.utils import resolve_device, resolve_dtype, set_seed, make_dataloader

logger = logging.getLogger(__name__)


class _TorchBaseEstimator(BaseEstimator, ClassifierMixin):
    """Shared logic for torch estimators."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        dropout: float = 0.0,
        hidden_dim: int = 128,
        n_layers: int = 1,
        patience: int = 10,
        seed: int = 42,
        device: str = "auto",
        torch_dtype: str = "float32",
        num_workers: int = 0,
        class_weight: str | dict[str, float] | None = "balanced",
        val_fraction: float = 0.1,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patience = patience
        self.seed = seed
        self.device = device
        self.torch_dtype = torch_dtype
        self.num_workers = num_workers
        self.class_weight = class_weight
        self.val_fraction = val_fraction

        self.model: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._input_dim: Optional[int] = None
        self._num_classes: Optional[int] = None
        self._is_fitted = False

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _create_loss(self, y_encoded: np.ndarray) -> nn.Module:
        raise NotImplementedError

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        return y_encoded

    def _prepare(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        y_encoded = self._encode_labels(y)
        self._input_dim = X.shape[1]
        self._num_classes = int(np.max(y_encoded)) + 1
        return X, y_encoded

    def _setup_device(self) -> None:
        resolved = resolve_device(self.device)
        self._device = torch.device(resolved)
        self._dtype = resolve_dtype(self.torch_dtype, resolved)
        logger.info(
            "%s using torch device=%s (requested=%s)",
            self.__class__.__name__,
            self._device,
            self.device,
        )

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        loss_fn: nn.Module,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_state = None
        best_metric = -np.inf
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loader = make_dataloader(
                X_train,
                y_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                worker_seed=self.seed,
            )
            for xb, yb in train_loader:
                xb = xb.to(self._device, dtype=self._dtype)
                yb = self._prepare_target(yb.to(self._device))
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            if X_val is None or y_val is None:
                continue

            self.model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_val).to(self._device, dtype=self._dtype)
                logits = self.model(xb)
                metric = self._validation_metric(logits, y_val)

            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.patience and epochs_no_improve >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _validation_metric(self, logits: torch.Tensor, y_val: np.ndarray) -> float:
        raise NotImplementedError

    def _prepare_target(self, yb: torch.Tensor) -> torch.Tensor:
        return yb

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchBaseEstimator":
        set_seed(self.seed)
        X, y_encoded = self._prepare(X, y)
        self._setup_device()

        self.model = self._build_model(self._input_dim, self._num_classes).to(self._device, dtype=self._dtype)
        try:
            param_device = next(self.model.parameters()).device
            logger.info("%s model parameters on device=%s", self.__class__.__name__, param_device)
        except StopIteration:
            logger.warning("%s model has no parameters to report device.", self.__class__.__name__)
        loss_fn = self._create_loss(y_encoded).to(self._device)

        X_train, X_val, y_train, y_val = None, None, None, None
        if self.val_fraction > 0 and len(np.unique(y_encoded)) > 1 and len(y_encoded) >= 10:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y_encoded,
                    test_size=self.val_fraction,
                    random_state=self.seed,
                    stratify=y_encoded,
                )
            except ValueError:
                X_train, y_train = X, y_encoded
        else:
            X_train, y_train = X, y_encoded

        self._train_loop(X_train, y_train, X_val, y_val, loss_fn)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.model is None:
            raise ValueError("Estimator is not fitted yet.")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("model", None)
        if model is not None:
            state["_model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            state["_model_state_dict"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        model_state = state.get("_model_state_dict")
        if model_state is not None and self._input_dim is not None and self._num_classes is not None:
            self._setup_device()
            self.model = self._build_model(self._input_dim, self._num_classes).to(self._device, dtype=self._dtype)
            self.model.load_state_dict(model_state)
            self._is_fitted = True


class TorchLogisticRegressionClassifier(_TorchBaseEstimator):
    """Binary logistic regression using a single linear layer."""

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        return BinaryLinear(input_dim)

    def _create_loss(self, y_encoded: np.ndarray) -> nn.Module:
        pos_weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_encoded)
            if len(counts) > 1 and counts[1] > 0:
                pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32)
        elif isinstance(self.class_weight, dict):
            if 1 in self.class_weight and 0 in self.class_weight:
                pos_weight = torch.tensor([self.class_weight[1] / self.class_weight[0]], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _validation_metric(self, logits: torch.Tensor, y_val: np.ndarray) -> float:
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        return f1_score(y_val, preds, zero_division=0)

    def _prepare_target(self, yb: torch.Tensor) -> torch.Tensor:
        return yb.float()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self._device, dtype=self._dtype)
            logits = self.model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        preds = (proba[:, 1] >= 0.5).astype(int)
        if self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds)
        return preds


class TorchMLPClassifier(TorchLogisticRegressionClassifier):
    """Binary MLP classifier."""

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        return BinaryMLP(input_dim, self.hidden_dim, self.n_layers, self.dropout)


class TorchSoftmaxRegressionClassifier(_TorchBaseEstimator):
    """Multiclass softmax regression."""

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        return MulticlassLinear(input_dim, num_classes)

    def _create_loss(self, y_encoded: np.ndarray) -> nn.Module:
        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_encoded)
            weights = np.zeros_like(counts, dtype=np.float32)
            non_zero = counts > 0
            weights[non_zero] = counts.sum() / (counts[non_zero] * len(counts[non_zero]))
            weight = torch.tensor(weights, dtype=torch.float32)
        elif isinstance(self.class_weight, dict):
            weights = np.zeros(self._num_classes, dtype=np.float32)
            for cls, w in self.class_weight.items():
                if cls < len(weights):
                    weights[cls] = w
            weight = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)

    def _validation_metric(self, logits: torch.Tensor, y_val: np.ndarray) -> float:
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        return f1_score(y_val, preds, average="macro", zero_division=0)

    def _prepare_target(self, yb: torch.Tensor) -> torch.Tensor:
        return yb.long()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self._device, dtype=self._dtype)
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        if self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds)
        return preds


class TorchMLPMulticlassClassifier(TorchSoftmaxRegressionClassifier):
    """Multiclass MLP classifier."""

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        return MulticlassMLP(input_dim, num_classes, self.hidden_dim, self.n_layers, self.dropout)
