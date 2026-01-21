"""Estimator registry and parameter grids."""

from __future__ import annotations

from typing import Dict, Any, Optional

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)


def get_estimators(
    random_state: int = 42,
    max_iter: int = 10000,
    logreg_params: Optional[Dict[str, Any]] = None,
    resolved_device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get dictionary of estimators with consistent configurations.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility
    max_iter : int
        Maximum iterations for iterative estimators

    Returns
    -------
    estimators : Dict[str, Any]
        Estimator name -> sklearn estimator instance
    """
    base_logreg_params: Dict[str, Any] = {
        "penalty": "l1",
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if logreg_params:
        base_logreg_params.update(logreg_params)

    estimators: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(**base_logreg_params),
        "SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            probability=True,
        ),
        "RandomForest": RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    if resolved_device in {"mps", "cuda"}:
        try:
            from classiflow.models.torch_multiclass import TorchLinearClassifier, TorchMLPClassifier

            estimators["torch_linear"] = TorchLinearClassifier(
                lr=1e-3,
                weight_decay=0.0,
                epochs=25,
                batch_size=256,
                class_weight="balanced",
                random_state=random_state,
                device=resolved_device,
            )
            estimators["torch_mlp"] = TorchMLPClassifier(
                lr=1e-3,
                weight_decay=1e-4,
                epochs=25,
                batch_size=256,
                hidden_dim=128,
                n_layers=1,
                class_weight="balanced",
                random_state=random_state,
                device=resolved_device,
            )
        except Exception as exc:
            logger.warning("Torch estimators unavailable (%s). Skipping torch models.", exc)

    return estimators


def get_param_grids(resolved_device: Optional[str] = None) -> Dict[str, Dict[str, list]]:
    """
    Get hyperparameter grids for GridSearchCV.

    Returns
    -------
    param_grids : Dict[str, Dict[str, list]]
        Estimator name -> parameter grid
    """
    grids: Dict[str, Dict[str, list]] = {
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1, 10],
        },
        "SVM": {
            "clf__C": [0.01, 0.1, 1, 10],
        },
        "RandomForest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
        },
        "GradientBoosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1],
            "clf__max_depth": [3, 5],
        },
    }

    if resolved_device in {"mps", "cuda"}:
        grids["torch_linear"] = {
            "clf__lr": [1e-2, 1e-3],
            "clf__weight_decay": [0.0, 1e-4],
            "clf__epochs": [10, 25],
        }
        grids["torch_mlp"] = {
            "clf__hidden_dim": [128, 256],
            "clf__lr": [1e-3],
            "clf__weight_decay": [1e-4],
            "clf__epochs": [25],
            "clf__batch_size": [256, 512],
        }

    return grids
