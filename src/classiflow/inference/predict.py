"""Prediction engine for binary, meta, and hierarchical models."""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BinaryPredictor:
    """Predict using binary task models."""

    def __init__(
        self,
        pipes: Dict[str, Any],
        best_models: Dict[str, str],
        task_names: Optional[List[str]] = None,
    ):
        """
        Initialize binary predictor.

        Parameters
        ----------
        pipes : Dict[str, Pipeline]
            Trained pipelines {task__model: pipeline}
        best_models : Dict[str, str]
            Best model per task {task: model_name}
        task_names : Optional[List[str]]
            Subset of tasks to use (None = all)
        """
        self.pipes = pipes
        self.best_models = best_models
        self.task_names = task_names or list(best_models.keys())

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Run binary task predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns:
            - {task}_score: continuous score
            - {task}_pred: binary prediction
        """
        predictions = pd.DataFrame(index=X.index)

        for task in self.task_names:
            if task not in self.best_models:
                logger.warning(f"Task '{task}' not in best_models, skipping")
                continue

            model_name = self.best_models[task]
            key = f"{task}__{model_name}"

            if key not in self.pipes:
                logger.warning(f"Pipeline '{key}' not found, skipping task '{task}'")
                continue

            pipe = self.pipes[key]

            try:
                scores = self._get_scores(pipe, X)
                predictions[f"{task}_score"] = scores
                predictions[f"{task}_pred"] = self._threshold_scores(scores)
            except Exception as e:
                logger.error(f"Prediction failed for task '{task}': {e}")
                predictions[f"{task}_score"] = np.nan
                predictions[f"{task}_pred"] = np.nan

        return predictions

    @staticmethod
    def _get_scores(pipe, X) -> np.ndarray:
        """Extract scores from pipeline (proba or decision function)."""
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        elif hasattr(pipe, "decision_function"):
            return pipe.decision_function(X)
        else:
            raise ValueError("Pipeline has neither predict_proba nor decision_function")

    @staticmethod
    def _threshold_scores(scores: np.ndarray) -> np.ndarray:
        """Apply threshold to scores to get binary predictions."""
        smin, smax = float(np.nanmin(scores)), float(np.nanmax(scores))

        # If scores are in [0, 1] range, use 0.5 threshold
        if smin >= 0.0 and smax <= 1.0:
            threshold = 0.5
        else:
            # Decision function: use 0.0
            threshold = 0.0

        return (scores >= threshold).astype(int)


class MetaPredictor:
    """Predict using meta-classifier."""

    def __init__(
        self,
        meta_model: Any,
        meta_features: List[str],
        meta_classes: List[str],
        calibration_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize meta predictor.

        Parameters
        ----------
        meta_model : Any
            Trained meta-classifier
        meta_features : List[str]
            Ordered list of meta-feature names
        meta_classes : List[str]
            Ordered list of class names
        calibration_metadata : Optional[Dict[str, Any]]
            Calibration metadata captured during training
        """
        self.meta_model = meta_model
        self.meta_features = meta_features
        self.meta_classes = meta_classes
        self.calibration_metadata = calibration_metadata or {}

    def predict(self, binary_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Run meta-classifier predictions.

        Parameters
        ----------
        binary_predictions : pd.DataFrame
            Binary task scores (must contain meta_features columns)

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns:
            - predicted_label: predicted class
            - predicted_proba_{class}: probability for each class
            - predicted_proba: max probability
        """
        # Check for missing meta-features
        missing = set(self.meta_features) - set(binary_predictions.columns)
        if missing:
            raise ValueError(f"Missing meta-features in binary predictions: {missing}")

        # Extract meta-features in correct order
        X_meta = binary_predictions[self.meta_features].values

        # Fill any NaN values with 0
        X_meta = np.nan_to_num(X_meta, nan=0.0)

        # Predict
        y_pred = self.meta_model.predict(X_meta).astype(str)

        predictions = pd.DataFrame(index=binary_predictions.index)
        predictions["predicted_label"] = y_pred
        predictions["y_pred"] = y_pred

        # Probabilities
        if hasattr(self.meta_model, "predict_proba"):
            y_proba = self.meta_model.predict_proba(X_meta)

            # Use meta_classes if available, otherwise get from model
            classes = self.meta_classes if self.meta_classes else \
                     [str(c) for c in self.meta_model.classes_]

            if len(classes) == y_proba.shape[1]:
                for i, cls in enumerate(classes):
                    predictions[f"predicted_proba_{cls}"] = y_proba[:, i]
                    predictions[f"y_prob_{cls}"] = y_proba[:, i]

                predictions["predicted_proba"] = y_proba.max(axis=1)
                predictions["y_prob"] = predictions["predicted_proba"]
            else:
                logger.warning(
                    f"Class count mismatch: {len(classes)} classes but {y_proba.shape[1]} proba columns"
                )
                predictions["y_prob"] = np.nan

            raw_model = getattr(self.meta_model, "base_estimator_", self.meta_model)
            raw_proba = _safe_raw_proba(raw_model, X_meta)
            if raw_proba is not None and raw_proba.shape[1] == len(classes):
                raw_scores = np.max(raw_proba, axis=1)
                predictions["y_score_raw"] = raw_scores
                for i, cls in enumerate(classes):
                    predictions[f"y_score_raw_{cls}"] = raw_proba[:, i]
            else:
                predictions["y_score_raw"] = np.nan
        else:
            predictions["y_prob"] = np.nan
            predictions["y_score_raw"] = np.nan

        predictions["threshold_used"] = 0.5 if len(classes) == 2 else None
        predictions["calibration_method"] = self.calibration_metadata.get("method_used")
        predictions["calibration_enabled"] = self.calibration_metadata.get("enabled", False)
        predictions["calibration_cv"] = self.calibration_metadata.get("cv")
        predictions["calibration_bins"] = self.calibration_metadata.get("bins")
        warnings = self.calibration_metadata.get("warnings") or []
        predictions["calibration_warnings"] = "; ".join(map(str, warnings))

        return predictions


class HierarchicalPredictor:
    """Predict using hierarchical models (L1 -> L2 routing)."""

    def __init__(
        self,
        models: Dict[str, Any],
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Initialize hierarchical predictor.

        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary with L1 and L2 models
        config : Dict[str, Any]
            Configuration with class information
        device : str
            Device for inference
        """
        self.models = models
        self.config = config
        self.device = device

        self.l1_model = models["l1_model"]
        self.l2_models = models.get("l2_models", {})
        self.scaler = models["scaler"]
        self.l1_encoder = models["l1_encoder"]
        self.l2_encoders = models.get("l2_encoders", {})

        self.l1_classes = config["l1_classes"]
        self.l2_classes_per_branch = config.get("l2_classes_per_branch", {})

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Run hierarchical predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns:
            - predicted_label_L1: L1 prediction
            - predicted_label_L2: L2 prediction (if applicable)
            - predicted_label: final combined prediction (L1::L2 or L1)
            - predicted_proba_L1_{class}: L1 probabilities
            - predicted_proba_L2_{class}: L2 probabilities (if applicable)
        """
        X_scaled = self.scaler.transform(X.values)

        # L1 predictions
        y_l1_pred_enc = self.l1_model.predict(X_scaled)
        y_l1_pred = self.l1_encoder.inverse_transform(y_l1_pred_enc)
        y_l1_proba = self.l1_model.predict_proba(X_scaled)

        predictions = pd.DataFrame(index=X.index)
        predictions["predicted_label_L1"] = y_l1_pred

        # L1 probabilities
        for i, cls in enumerate(self.l1_classes):
            predictions[f"predicted_proba_L1_{cls}"] = y_l1_proba[:, i]

        # L2 predictions (if hierarchical)
        if self.config.get("hierarchical") and self.l2_models:
            l2_predictions = []
            pipeline_predictions = []

            for idx, l1_class in enumerate(y_l1_pred):
                if l1_class in self.l2_models and self.l2_models[l1_class] is not None:
                    # Predict L2 for this branch
                    model_l2 = self.l2_models[l1_class]
                    le_l2 = self.l2_encoders.get(l1_class)

                    X_sample = X_scaled[idx:idx+1]
                    y_l2_pred_enc = model_l2.predict(X_sample)[0]

                    if le_l2 is not None:
                        y_l2_pred = le_l2.inverse_transform([y_l2_pred_enc])[0]
                    else:
                        l2_classes = self.l2_classes_per_branch.get(l1_class, [])
                        y_l2_pred = l2_classes[y_l2_pred_enc] if y_l2_pred_enc < len(l2_classes) else "Unknown"

                    l2_predictions.append(y_l2_pred)
                    pipeline_predictions.append(f"{l1_class}::{y_l2_pred}")
                else:
                    l2_predictions.append(None)
                    pipeline_predictions.append(l1_class)

            predictions["predicted_label_L2"] = l2_predictions
            predictions["predicted_label"] = pipeline_predictions
        else:
            predictions["predicted_label"] = y_l1_pred

        return predictions


class MulticlassPredictor:
    """Predict using a direct multiclass estimator."""

    def __init__(self, model: Any, classes: Optional[List[str]] = None):
        self.model = model
        self.classes = classes or getattr(model, "classes_", None)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(index=X.index)
        y_pred = self.model.predict(X)
        predictions["predicted_label"] = y_pred

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)
            classes = self.classes or [str(i) for i in range(y_proba.shape[1])]
            for idx, cls in enumerate(classes):
                predictions[f"predicted_proba_{cls}"] = y_proba[:, idx]
            predictions["predicted_proba"] = y_proba.max(axis=1)

        return predictions


def _safe_raw_proba(model, X):
    """Attempt to compute probabilities without altering state."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception as exc:
            logger.warning(f"Raw predict_proba failed: {exc}")
    return None
