"""Backend registry for estimator sets and parameter grids."""

from __future__ import annotations

from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression

from classiflow.models.estimators import get_estimators as get_sklearn_estimators
from classiflow.models.estimators import get_param_grids as get_sklearn_param_grids
from classiflow.backends.torch.estimators import (
    TorchLogisticRegressionClassifier,
    TorchMLPClassifier,
    TorchSoftmaxRegressionClassifier,
    TorchMLPMulticlassClassifier,
)


def get_backend(name: str | None) -> str:
    """Normalize backend name."""
    if not name:
        return "sklearn"
    name = name.lower()
    if name not in {"sklearn", "torch"}:
        raise ValueError(f"Unsupported backend: {name}")
    return name


def _default_model_set(backend: str) -> str:
    if backend == "torch":
        return "torch_basic"
    return "default"


def get_model_set(
    command: str,
    backend: str,
    model_set: str | None,
    *,
    random_state: int = 42,
    max_iter: int = 10000,
    device: str = "auto",
    torch_dtype: str = "float32",
    torch_num_workers: int = 0,
    meta_C_grid: list[float] | None = None,
) -> Dict[str, Any]:
    """
    Resolve estimator sets and parameter grids for a command/backend/model_set.

    Returns
    -------
    Dict[str, Any]
        For train-binary:
          {"estimators": Dict[str, Any], "param_grids": Dict[str, Dict[str, list]]}
        For train-meta:
          {
            "base_estimators": Dict[str, Any],
            "base_param_grids": Dict[str, Dict[str, list]],
            "meta_estimators": Dict[str, Any],
            "meta_param_grids": Dict[str, Dict[str, list]],
          }
    """
    backend = get_backend(backend)
    model_set = model_set or _default_model_set(backend)

    if command not in {"train-binary", "train-meta"}:
        raise ValueError(f"Unsupported command for model registry: {command}")

    if backend == "sklearn":
        base_estimators = get_sklearn_estimators(random_state, max_iter)
        base_param_grids = get_sklearn_param_grids()

        if model_set != "default":
            raise ValueError(f"Unsupported sklearn model_set: {model_set}")

        if command == "train-binary":
            return {"estimators": base_estimators, "param_grids": base_param_grids}

        meta_grid = {"C": meta_C_grid or [0.01, 0.1, 1, 10]}
        meta_estimators = {
            "MultinomialLogReg": LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
            )
        }
        meta_param_grids = {"MultinomialLogReg": meta_grid}
        return {
            "base_estimators": base_estimators,
            "base_param_grids": base_param_grids,
            "meta_estimators": meta_estimators,
            "meta_param_grids": meta_param_grids,
        }

    if model_set not in {"torch_basic", "torch_fast"}:
        raise ValueError(f"Unsupported torch model_set: {model_set}")

    if command == "train-binary":
        estimators = {
            "TorchLogisticRegression": TorchLogisticRegressionClassifier(
                lr=1e-3,
                weight_decay=0.0,
                epochs=200 if model_set == "torch_basic" else 100,
                batch_size=256,
                patience=10,
                seed=random_state,
                device=device,
                torch_dtype=torch_dtype,
                num_workers=torch_num_workers,
                class_weight="balanced",
            ),
            "TorchMLP": TorchMLPClassifier(
                lr=1e-3,
                weight_decay=1e-4,
                epochs=200 if model_set == "torch_basic" else 100,
                batch_size=256,
                dropout=0.3,
                hidden_dim=128,
                n_layers=2,
                patience=10,
                seed=random_state,
                device=device,
                torch_dtype=torch_dtype,
                num_workers=torch_num_workers,
                class_weight="balanced",
            ),
        }
        if model_set == "torch_fast":
            grids = {
                "TorchLogisticRegression": {
                    "clf__lr": [1e-3],
                    "clf__weight_decay": [0.0],
                    "clf__epochs": [10],
                },
                "TorchMLP": {
                    "clf__lr": [1e-3],
                    "clf__weight_decay": [1e-4],
                    "clf__hidden_dim": [64],
                    "clf__n_layers": [1],
                    "clf__dropout": [0.2],
                    "clf__epochs": [10],
                },
            }
        else:
            grids = {
                "TorchLogisticRegression": {
                    "clf__lr": [5e-4, 1e-3, 5e-3],
                    "clf__weight_decay": [0.0, 1e-4, 1e-3],
                    "clf__epochs": [100, 200],
                },
                "TorchMLP": {
                    "clf__lr": [5e-4, 1e-3],
                    "clf__weight_decay": [1e-4, 1e-3],
                    "clf__hidden_dim": [64, 128],
                    "clf__n_layers": [1, 2, 3],
                    "clf__dropout": [0.2, 0.4],
                    "clf__epochs": [100, 200],
                },
            }
        return {"estimators": estimators, "param_grids": grids}

    base = {
        "TorchLogisticRegression": TorchLogisticRegressionClassifier(
            lr=1e-3,
            weight_decay=0.0,
            epochs=200 if model_set == "torch_basic" else 100,
            batch_size=256,
            patience=10,
            seed=random_state,
            device=device,
            torch_dtype=torch_dtype,
            num_workers=torch_num_workers,
            class_weight="balanced",
        ),
        "TorchMLP": TorchMLPClassifier(
            lr=1e-3,
            weight_decay=1e-4,
            epochs=200 if model_set == "torch_basic" else 100,
            batch_size=256,
            dropout=0.3,
            hidden_dim=128,
            n_layers=2,
            patience=10,
            seed=random_state,
            device=device,
            torch_dtype=torch_dtype,
            num_workers=torch_num_workers,
            class_weight="balanced",
        ),
    }
    if model_set == "torch_fast":
        base_grids = {
            "TorchLogisticRegression": {
                "clf__lr": [1e-3],
                "clf__weight_decay": [0.0],
                "clf__epochs": [10],
            },
            "TorchMLP": {
                "clf__lr": [1e-3],
                "clf__weight_decay": [1e-4],
                "clf__hidden_dim": [64],
                "clf__n_layers": [1],
                "clf__dropout": [0.2],
                "clf__epochs": [10],
            },
        }
    else:
        base_grids = {
            "TorchLogisticRegression": {
                "clf__lr": [5e-4, 1e-3, 5e-3],
                "clf__weight_decay": [0.0, 1e-4, 1e-3],
                "clf__epochs": [100, 200],
            },
            "TorchMLP": {
                "clf__lr": [5e-4, 1e-3],
                "clf__weight_decay": [1e-4, 1e-3],
                "clf__hidden_dim": [64, 128],
                "clf__n_layers": [1, 2, 3],
                "clf__dropout": [0.2, 0.4],
                "clf__epochs": [100, 200],
            },
        }

    meta_estimators = {
        "TorchSoftmaxRegression": TorchSoftmaxRegressionClassifier(
            lr=1e-3,
            weight_decay=0.0,
            epochs=200 if model_set == "torch_basic" else 100,
            batch_size=256,
            patience=10,
            seed=random_state,
            device=device,
            torch_dtype=torch_dtype,
            num_workers=torch_num_workers,
            class_weight="balanced",
        ),
        "TorchMLPMulticlass": TorchMLPMulticlassClassifier(
            lr=1e-3,
            weight_decay=1e-4,
            epochs=200 if model_set == "torch_basic" else 100,
            batch_size=256,
            dropout=0.3,
            hidden_dim=128,
            n_layers=2,
            patience=10,
            seed=random_state,
            device=device,
            torch_dtype=torch_dtype,
            num_workers=torch_num_workers,
            class_weight="balanced",
        ),
    }

    if model_set == "torch_fast":
        meta_grids = {
            "TorchSoftmaxRegression": {
                "lr": [1e-3],
                "weight_decay": [0.0],
                "epochs": [10],
            },
            "TorchMLPMulticlass": {
                "lr": [1e-3],
                "weight_decay": [1e-4],
                "hidden_dim": [64],
                "n_layers": [1],
                "dropout": [0.2],
                "epochs": [10],
            },
        }
    else:
        meta_grids = {
            "TorchSoftmaxRegression": {
                "lr": [5e-4, 1e-3, 5e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "epochs": [100, 200],
            },
            "TorchMLPMulticlass": {
                "lr": [5e-4, 1e-3],
                "weight_decay": [1e-4, 1e-3],
                "hidden_dim": [64, 128],
                "n_layers": [1, 2, 3],
                "dropout": [0.2, 0.4],
                "epochs": [100, 200],
            },
        }

    return {
        "base_estimators": base,
        "base_param_grids": base_grids,
        "meta_estimators": meta_estimators,
        "meta_param_grids": meta_grids,
    }
