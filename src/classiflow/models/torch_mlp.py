"""PyTorch MLP implementation with device support (CPU/CUDA/MPS)."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> str:
    """
    Resolve device name to available hardware.

    Parameters
    ----------
    name : str
        Device name: "auto", "cpu", "cuda", or "mps"

    Returns
    -------
    str
        Resolved device name

    Notes
    -----
    - "auto" prioritizes: CUDA -> MPS -> CPU
    - Falls back to CPU if requested device unavailable
    """
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if name == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    return name


class TorchMLP(nn.Module):
    """
    Flexible multi-layer perceptron with configurable architecture.

    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : List[int]
        Hidden layer dimensions
    num_classes : int
        Number of output classes
    dropout : float
        Dropout probability

    Examples
    --------
    >>> model = TorchMLP(input_dim=100, hidden_dims=[128, 64], num_classes=3, dropout=0.3)
    >>> x = torch.randn(32, 100)  # batch of 32 samples
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([32, 3])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.net(x)


class TorchMLPWrapper:
    """
    Scikit-learn compatible wrapper for PyTorch MLP with early stopping.

    Parameters
    ----------
    input_dim : int
        Number of input features
    num_classes : int
        Number of output classes
    hidden_dims : List[int], optional
        Hidden layer dimensions (default: [128])
    lr : float
        Learning rate (default: 1e-3)
    weight_decay : float
        L2 regularization weight (default: 1e-5)
    batch_size : int
        Training batch size (default: 256)
    epochs : int
        Maximum training epochs (default: 100)
    dropout : float
        Dropout probability (default: 0.3)
    early_stopping_patience : int
        Early stopping patience (default: 10)
    device : str
        Device: "auto", "cpu", "cuda", or "mps" (default: "cpu")
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level (0=silent, 1=standard, 2=detailed)

    Examples
    --------
    >>> wrapper = TorchMLPWrapper(
    ...     input_dim=100,
    ...     num_classes=3,
    ...     hidden_dims=[128, 64],
    ...     device="auto",
    ...     random_state=42
    ... )
    >>> wrapper.fit(X_train, y_train, X_val, y_val)
    >>> y_pred = wrapper.predict(X_test)
    >>> y_proba = wrapper.predict_proba(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 100,
        dropout: float = 0.3,
        early_stopping_patience: int = 10,
        device: str = "cpu",
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [128]
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.device_name = device
        self.random_state = random_state
        self.verbose = verbose

        self._device = torch.device(resolve_device(device))
        self.model = TorchMLP(
            input_dim, self.hidden_dims, num_classes, dropout
        ).to(self._device)

        self.best_state_dict = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_f1": []}
        self.best_epoch = 0

        if verbose >= 1:
            logger.info(f"Initialized TorchMLPWrapper on device: {self._device}")

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[np.ndarray] = None,
    ) -> TorchMLPWrapper:
        """
        Train model with optional early stopping on validation set.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels (0-indexed integers)
        X_val : np.ndarray, optional
            Validation features for early stopping
        y_val : np.ndarray, optional
            Validation labels
        class_weights : np.ndarray, optional
            Class weights for imbalanced data

        Returns
        -------
        self
            Fitted model
        """
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Re-initialize model
        self.model = TorchMLP(
            self.input_dim, self.hidden_dims, self.num_classes, self.dropout
        ).to(self._device)

        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        # Set up loss with optional class weights
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self._device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches
            self.training_history["train_loss"].append(avg_train_loss)

            # Validation phase (if validation data provided)
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self._device)
                    y_val_t = torch.tensor(y_val, dtype=torch.int64).to(self._device)

                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()

                    val_preds = val_logits.argmax(dim=1).cpu().numpy()
                    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_f1"].append(val_f1)

                scheduler.step(val_loss)

                if self.verbose >= 2:
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.epochs}: "
                        f"train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_f1={val_f1:.4f}"
                    )

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state_dict = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.best_epoch = epoch + 1
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if self.verbose >= 2:
                        logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if self.verbose >= 2:
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.epochs}: train_loss={avg_train_loss:.4f}"
                    )

        # Restore best model if early stopping was used
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(self._device)
            if self.verbose >= 1:
                logger.info(f"Restored best model from epoch {self.best_epoch}")

        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        """Get model logits for input data."""
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        # Process in batches for large datasets
        all_logits = []
        n_samples = len(X)

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch = X[i : i + self.batch_size]
                xb = torch.from_numpy(batch).to(self._device)
                logits = self.model(xb).cpu().numpy()
                all_logits.append(logits)

        return np.vstack(all_logits)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_samples, n_classes)
        """
        logits = self._logits(X)
        # Softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features

        Returns
        -------
        np.ndarray
            Predicted class indices
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path: Path) -> None:
        """Save model state dict to file."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load model state dict from file."""
        self.model.load_state_dict(torch.load(path, map_location=self._device))
        self.model.to(self._device)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device_name,
            "best_epoch": self.best_epoch,
        }
