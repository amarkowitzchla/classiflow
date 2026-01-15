"""Artifact loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model with metadata.

    Parameters
    ----------
    path : Path
        Path to .joblib file

    Returns
    -------
    model : Any
        Loaded model
    metadata : Dict
        Metadata dictionary (empty if not present)
    """
    payload = joblib.load(path)

    if isinstance(payload, dict):
        model = payload.get("model")
        metadata = payload.get("metadata", {})
    else:
        # Old format: just the model
        model = payload
        metadata = {}

    logger.info(f"Loaded model from {path}")
    return model, metadata


def load_meta_pipeline(fold_dir: Path, variant: str = "smote") -> Dict[str, Any]:
    """
    Load full meta-classifier pipeline artifacts.

    Parameters
    ----------
    fold_dir : Path
        Fold directory (e.g., derived/fold1)
    variant : str
        SMOTE variant ("smote" or "none")

    Returns
    -------
    pipeline : Dict[str, Any]
        Contains: binary_pipes, best_models, meta_model, meta_features, meta_classes
    """
    var_dir = fold_dir / f"binary_{variant}"

    # Binary pipes
    binary_data = joblib.load(var_dir / "binary_pipes.joblib")
    binary_pipes = binary_data["pipes"]
    best_models = binary_data["best_models"]

    # Meta model
    meta_model = joblib.load(var_dir / "meta_model.joblib")

    # Meta features
    meta_features = pd.read_csv(var_dir / "meta_features.csv", header=None)[0].tolist()

    # Meta classes
    meta_classes = pd.read_csv(var_dir / "meta_classes.csv", header=None)[0].tolist()

    logger.info(f"Loaded meta pipeline from {fold_dir} (variant={variant})")

    return {
        "binary_pipes": binary_pipes,
        "best_models": best_models,
        "meta_model": meta_model,
        "meta_features": meta_features,
        "meta_classes": meta_classes,
    }
