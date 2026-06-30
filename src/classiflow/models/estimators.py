"""Estimator registry and parameter grids."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from classiflow.config import default_torch_num_workers
from classiflow.models.ensemble import (
    adapt_param_grid_for_strategy,
    wrap_estimator_for_strategy,
)
from classiflow.models.mlp_tuning import build_torch_mlp_param_grid, prefix_param_grid

logger = logging.getLogger(__name__)


def get_estimators(
    random_state: int = 42,
    max_iter: int = 10000,
    logreg_params: Optional[Dict[str, Any]] = None,
    resolved_device: Optional[str] = None,
    torch_num_workers: int | None = None,
    torch_temperature_scaling: bool = False,
    expanded_mlp_tuning_grid: bool = False,
    final_estimator_strategy: str = "single",
    bagging_n_estimators: int = 10,
    bagging_max_samples: float = 1.0,
    bagging_max_features: float = 1.0,
    bagging_bootstrap: bool = True,
    bagging_bootstrap_features: bool = False,
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
        # penalty deprecated in sklearn 1.8; prefer l1_ratio/C if tuning
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if logreg_params:
        base_logreg_params.update(logreg_params)
    resolved_torch_workers = (
        default_torch_num_workers() if torch_num_workers is None else torch_num_workers
    )

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
                num_workers=resolved_torch_workers,
                temperature_scaling=torch_temperature_scaling,
            )
            estimators["torch_mlp"] = TorchMLPClassifier(
                lr=1e-3,
                weight_decay=1e-4,
                epochs=25,
                batch_size=256,
                hidden_dim=128,
                n_layers=1,
                dropout=0.0,
                activation="relu",
                use_batchnorm=False,
                class_weight="balanced",
                random_state=random_state,
                device=resolved_device,
                num_workers=resolved_torch_workers,
                temperature_scaling=torch_temperature_scaling,
            )
        except Exception as exc:
            logger.warning("Torch estimators unavailable (%s). Skipping torch models.", exc)

    return {
        name: wrap_estimator_for_strategy(
            estimator,
            strategy=final_estimator_strategy,
            random_state=random_state,
            bagging_n_estimators=bagging_n_estimators,
            bagging_max_samples=bagging_max_samples,
            bagging_max_features=bagging_max_features,
            bagging_bootstrap=bagging_bootstrap,
            bagging_bootstrap_features=bagging_bootstrap_features,
        )
        for name, estimator in estimators.items()
    }


def get_param_grids(
    resolved_device: Optional[str] = None,
    *,
    expanded_mlp_tuning_grid: bool = False,
    final_estimator_strategy: str = "single",
) -> Dict[str, Dict[str, list]]:
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
        grids["torch_mlp"] = prefix_param_grid(
            build_torch_mlp_param_grid("basic", expanded=expanded_mlp_tuning_grid),
            prefix="clf__",
        )

    return {
        name: adapt_param_grid_for_strategy(
            grid,
            strategy=final_estimator_strategy,
            pipeline_prefix="clf__",
        )
        for name, grid in grids.items()
    }
