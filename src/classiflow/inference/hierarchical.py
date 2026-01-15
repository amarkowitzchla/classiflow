"""Inference with trained hierarchical models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import torch

from classiflow.models.torch_mlp import TorchMLPWrapper

logger = logging.getLogger(__name__)


class HierarchicalInference:
    """
    Load and run inference with trained hierarchical models.

    Parameters
    ----------
    model_dir : Path
        Directory containing trained models (output from train_hierarchical)
    fold : int
        Which fold's models to use (1-indexed)
    device : str
        Device for inference: "auto", "cpu", "cuda", or "mps"

    Attributes
    ----------
    config : Dict
        Training configuration
    fold : int
        Selected fold
    device : str
        Inference device
    scaler : StandardScaler
        Feature scaler
    le_l1 : LabelEncoder
        Level-1 label encoder
    model_l1 : TorchMLPWrapper
        Level-1 model
    hierarchical : bool
        Whether models are hierarchical
    branch_models : Dict
        Level-2 models per branch (if hierarchical)
    branch_encoders : Dict
        Level-2 label encoders per branch (if hierarchical)
    l1_classes : List[str]
        Level-1 class names
    l2_classes_per_branch : Dict
        Level-2 class names per branch (if hierarchical)

    Examples
    --------
    >>> infer = HierarchicalInference("results/", fold=1, device="auto")
    >>> predictions = infer.predict(X_test)
    >>> predictions_proba = infer.predict_proba(X_test)
    """

    def __init__(
        self,
        model_dir: Path,
        fold: int = 1,
        device: str = "auto",
    ):
        self.model_dir = Path(model_dir)
        self.fold = fold
        self.device = device

        # Load configuration
        config_path = self.model_dir / "training_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        self.hierarchical = self.config.get("label_l2") is not None

        # Load fold-specific artifacts
        self.fold_dir = self.model_dir / f"fold{fold}"
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {self.fold_dir}")

        logger.info(f"Loading models from {self.fold_dir}")

        # Load scaler
        self.scaler = joblib.load(self.fold_dir / "scaler.joblib")

        # Load L1 encoder and model
        self.le_l1 = joblib.load(self.fold_dir / "label_encoder_l1.joblib")
        self.l1_classes = self.le_l1.classes_.tolist()

        self.model_l1 = self._load_model_l1()

        # Load L2 models if hierarchical
        if self.hierarchical:
            self.branch_models = {}
            self.branch_encoders = {}
            self.l2_classes_per_branch = {}

            for l1_class in self.l1_classes:
                safe_l1 = l1_class.replace(" ", "_")
                encoder_path = self.fold_dir / f"label_encoder_l2_{safe_l1}.joblib"
                model_path = self.fold_dir / f"model_level2_{safe_l1}_fold{fold}.pt"

                if encoder_path.exists() and model_path.exists():
                    le_l2 = joblib.load(encoder_path)
                    model_l2 = self._load_model_l2(model_path, len(le_l2.classes_))

                    self.branch_models[l1_class] = model_l2
                    self.branch_encoders[l1_class] = le_l2
                    self.l2_classes_per_branch[l1_class] = le_l2.classes_.tolist()

                    logger.debug(f"Loaded L2 model for branch: {l1_class}")
        else:
            self.branch_models = None
            self.branch_encoders = None
            self.l2_classes_per_branch = None

        logger.info(
            f"Loaded {'hierarchical' if self.hierarchical else 'single-label'} model"
        )
        logger.info(f"L1 classes: {self.l1_classes}")
        if self.hierarchical:
            logger.info(f"L2 branches: {list(self.branch_models.keys())}")

    def _load_model_l1(self) -> TorchMLPWrapper:
        """Load Level-1 model."""
        config_path = self.fold_dir / f"model_config_l1_fold{self.fold}.json"
        with open(config_path) as f:
            model_config = json.load(f)

        model = TorchMLPWrapper(
            input_dim=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            hidden_dims=model_config["hidden_dims"],
            dropout=model_config["dropout"],
            device=self.device,
            verbose=0,
        )

        model_path = self.fold_dir / f"model_level1_fold{self.fold}.pt"
        model.load(model_path)

        logger.debug(f"Loaded L1 model from {model_path}")
        return model

    def _load_model_l2(self, model_path: Path, num_classes: int) -> TorchMLPWrapper:
        """Load Level-2 branch model."""
        # Extract config from path
        # model_level2_TypeA_fold1.pt -> model_config_l2_TypeA_fold1.json
        config_path = model_path.parent / model_path.name.replace(
            "model_level2_", "model_config_l2_"
        ).replace(
            ".pt", ".json"
        )

        # If config doesn't exist, use L1 config as template
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            # Fall back to L1 config structure
            config_l1_path = self.fold_dir / f"model_config_l1_fold{self.fold}.json"
            with open(config_l1_path) as f:
                model_config = json.load(f)
            model_config["num_classes"] = num_classes

        model = TorchMLPWrapper(
            input_dim=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            hidden_dims=model_config["hidden_dims"],
            dropout=model_config["dropout"],
            device=self.device,
            verbose=0,
        )

        model.load(model_path)
        logger.debug(f"Loaded L2 model from {model_path}")
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted labels (strings)
            - Single-label mode: L1 predictions
            - Hierarchical mode: Combined L1::L2 predictions

        Examples
        --------
        >>> predictions = infer.predict(X_test)
        >>> predictions[0]
        'TypeA::SubtypeX'
        """
        X_scaled = self.scaler.transform(X)

        # Predict L1
        y_l1_pred_enc = self.model_l1.predict(X_scaled)
        y_l1_pred = self.le_l1.inverse_transform(y_l1_pred_enc)

        if not self.hierarchical:
            return y_l1_pred

        # Hierarchical mode: route to L2 branches
        predictions = []

        for i, l1_class in enumerate(y_l1_pred):
            if l1_class in self.branch_models:
                # Predict L2
                model_l2 = self.branch_models[l1_class]
                le_l2 = self.branch_encoders[l1_class]

                y_l2_pred_enc = model_l2.predict(X_scaled[i:i+1])[0]
                y_l2_pred = le_l2.inverse_transform([y_l2_pred_enc])[0]

                predictions.append(f"{l1_class}::{y_l2_pred}")
            else:
                # No trained L2 model for this branch
                predictions.append(f"{l1_class}::NA")

        return np.array(predictions)

    def predict_proba(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n_samples, n_features)

        Returns
        -------
        Tuple[np.ndarray, Optional[Dict]]
            - L1 probabilities: shape (n_samples, n_l1_classes)
            - L2 probabilities: Dict[l1_class, proba] if hierarchical, else None

        Examples
        --------
        >>> l1_proba, l2_proba = infer.predict_proba(X_test)
        >>> l1_proba.shape
        (100, 3)
        >>> l2_proba["TypeA"].shape
        (40, 5)  # 40 samples predicted as TypeA, 5 L2 classes
        """
        X_scaled = self.scaler.transform(X)

        # Get L1 probabilities
        y_l1_proba = self.model_l1.predict_proba(X_scaled)

        if not self.hierarchical:
            return y_l1_proba, None

        # Get L2 probabilities per branch
        y_l1_pred_enc = self.model_l1.predict(X_scaled)
        y_l1_pred = self.le_l1.inverse_transform(y_l1_pred_enc)

        l2_probas = {}

        for l1_class in self.l1_classes:
            if l1_class not in self.branch_models:
                continue

            # Find samples predicted as this L1 class
            mask = (y_l1_pred == l1_class)
            if mask.sum() == 0:
                continue

            model_l2 = self.branch_models[l1_class]
            l2_probas[l1_class] = model_l2.predict_proba(X_scaled[mask])

        return y_l1_proba, l2_probas

    def predict_dataframe(
        self, X: np.ndarray, include_proba: bool = True
    ) -> pd.DataFrame:
        """
        Predict and return results as DataFrame.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n_samples, n_features)
        include_proba : bool
            Whether to include probability columns

        Returns
        -------
        pd.DataFrame
            Predictions with optional probabilities

        Examples
        --------
        >>> df_pred = infer.predict_dataframe(X_test, include_proba=True)
        >>> df_pred.columns
        ['prediction', 'l1_class', 'l2_class', 'l1_proba_TypeA', 'l1_proba_TypeB', ...]
        """
        predictions = self.predict(X)

        results = {"prediction": predictions}

        # Split hierarchical predictions
        if self.hierarchical:
            l1_parts = []
            l2_parts = []
            for pred in predictions:
                if "::" in pred:
                    l1, l2 = pred.split("::", 1)
                    l1_parts.append(l1)
                    l2_parts.append(l2)
                else:
                    l1_parts.append(pred)
                    l2_parts.append("NA")

            results["l1_class"] = l1_parts
            results["l2_class"] = l2_parts
        else:
            results["l1_class"] = predictions

        # Add probabilities
        if include_proba:
            y_l1_proba, y_l2_proba = self.predict_proba(X)

            # L1 probabilities
            for i, cls in enumerate(self.l1_classes):
                results[f"l1_proba_{cls}"] = y_l1_proba[:, i]

            # L2 probabilities (if hierarchical)
            if self.hierarchical and y_l2_proba:
                # Initialize all L2 probability columns with NaN
                for l1_class in self.l2_classes_per_branch:
                    l2_classes = self.l2_classes_per_branch[l1_class]
                    for l2_cls in l2_classes:
                        col_name = f"l2_proba_{l1_class}_{l2_cls}"
                        results[col_name] = [np.nan] * len(X)

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Fill in L2 probabilities if hierarchical
        if include_proba and self.hierarchical and y_l2_proba:
            y_l1_pred_enc = self.model_l1.predict(self.scaler.transform(X))
            y_l1_pred = self.le_l1.inverse_transform(y_l1_pred_enc)

            for l1_class, l2_proba in y_l2_proba.items():
                mask = (y_l1_pred == l1_class)
                if mask.sum() > 0:
                    l2_classes = self.l2_classes_per_branch[l1_class]
                    for i, l2_cls in enumerate(l2_classes):
                        col_name = f"l2_proba_{l1_class}_{l2_cls}"
                        df_results.loc[mask, col_name] = l2_proba[:, i]

        return df_results

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            "fold": self.fold,
            "device": self.device,
            "hierarchical": self.hierarchical,
            "l1_classes": self.l1_classes,
            "l2_classes_per_branch": self.l2_classes_per_branch,
            "n_features": self.scaler.n_features_in_,
        }
