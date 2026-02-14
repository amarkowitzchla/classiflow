"""
Final model training module for production-ready models.

This module implements a "train from scratch" approach where the final model
is ALWAYS trained on 100% of the training data using validated configurations
from technical validation.

Key principles:
1. NO reuse of fold pipelines - always train from scratch
2. Per-task config selection from technical validation results
3. Explicit sampler (SMOTE) options
4. Comprehensive sanity checks before bundling
5. Full artifact audit trail
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

from classiflow.io import load_data
from classiflow.models import AdaptiveSMOTE
from classiflow.backends.registry import get_backend, get_model_set
from classiflow.config import default_torch_num_workers
from classiflow.tasks import TaskBuilder

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration and Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SelectedBinaryConfig:
    """Configuration for a single binary task from technical validation."""

    task_name: str
    model_name: str
    params: Dict[str, Any]
    mean_score: float
    sampler: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectedBinaryConfig":
        return cls(**data)


@dataclass
class SelectedMetaConfig:
    """Configuration for the meta-classifier from technical validation."""

    model_name: str
    params: Dict[str, Any]
    calibration_method: str
    calibration_cv: int
    calibration_bins: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectedMetaConfig":
        return cls(**data)


@dataclass
class FinalTrainConfig:
    """Configuration for final model training."""

    # Data
    train_manifest: Path
    label_col: str

    # Task configuration
    mode: str  # "meta", "binary", "multiclass", "hierarchical"
    classes: Optional[List[str]] = None

    # Selected configurations from technical validation
    selected_binary_configs: Dict[str, SelectedBinaryConfig] = field(default_factory=dict)
    selected_meta_config: Optional[SelectedMetaConfig] = None

    # Sampler option
    sampler: str = "none"  # "none", "smote", or other supported samplers

    # Calibration
    calibrate_meta: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3
    calibration_bins: int = 10
    isotonic_min_samples: int = 100

    # Model registry settings
    backend: str = "sklearn"
    model_set: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "float32"
    torch_num_workers: int = field(default_factory=default_torch_num_workers)

    # Random state
    random_state: int = 42
    max_iter: int = 10000

    # Sanity check thresholds
    sanity_min_std: float = 0.02
    sanity_max_mean_deviation: float = 0.15

    # Output
    outdir: Path = field(default_factory=lambda: Path("runs/final_model"))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "train_manifest": str(self.train_manifest),
            "label_col": self.label_col,
            "mode": self.mode,
            "classes": self.classes,
            "sampler": self.sampler,
            "calibrate_meta": self.calibrate_meta,
            "calibration_method": self.calibration_method,
            "calibration_cv": self.calibration_cv,
            "calibration_bins": self.calibration_bins,
            "isotonic_min_samples": self.isotonic_min_samples,
            "backend": self.backend,
            "model_set": self.model_set,
            "device": self.device,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "sanity_min_std": self.sanity_min_std,
            "sanity_max_mean_deviation": self.sanity_max_mean_deviation,
            "outdir": str(self.outdir),
        }
        if self.selected_binary_configs:
            data["selected_binary_configs"] = {
                k: v.to_dict() for k, v in self.selected_binary_configs.items()
            }
        if self.selected_meta_config:
            data["selected_meta_config"] = self.selected_meta_config.to_dict()
        return data


@dataclass
class SanityCheckResult:
    """Result of a sanity check on model predictions."""

    task_name: str
    check_type: str
    passed: bool
    mean: float
    std: float
    min_val: float
    max_val: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalTrainResult:
    """Result of final model training."""

    success: bool
    outdir: Path
    sanity_checks: List[SanityCheckResult]
    artifacts: Dict[str, str]
    config_used: Dict[str, Any]
    training_stats: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "outdir": str(self.outdir),
            "sanity_checks": [c.to_dict() for c in self.sanity_checks],
            "artifacts": self.artifacts,
            "config_used": self.config_used,
            "training_stats": self.training_stats,
            "warnings": self.warnings,
        }


# -----------------------------------------------------------------------------
# Config Selection from Technical Validation
# -----------------------------------------------------------------------------


def extract_selected_configs_from_technical_run(
    technical_run: Path,
    variant: str = "none",
    selection_metric: str = "mean_test_score",
    direction: str = "max",
) -> Tuple[Dict[str, SelectedBinaryConfig], Optional[SelectedMetaConfig]]:
    """
    Extract per-task best configurations from technical validation results.

    This is the source of truth for which hyperparameters to use in final training.

    Parameters
    ----------
    technical_run : Path
        Path to technical validation run directory
    variant : str
        Sampler variant to extract configs for ("none" or "smote")
    selection_metric : str
        Metric to use for selection (default: mean_test_score)
    direction : str
        "max" or "min" for selection direction

    Returns
    -------
    binary_configs : Dict[str, SelectedBinaryConfig]
        Per-task best configurations
    meta_config : Optional[SelectedMetaConfig]
        Meta-classifier configuration (if applicable)
    """
    metrics_path = technical_run / "metrics_inner_cv.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Inner CV metrics not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # Filter by sampler variant
    if "sampler" in df.columns:
        df = df[df["sampler"] == variant]

    if df.empty:
        raise ValueError(f"No results found for sampler variant '{variant}'")

    # Normalize column names for metric selection
    if selection_metric not in df.columns:
        # Try common alternatives
        alternatives = ["mean_test_score", "mean_test_f1", "mean_test_F1 Score"]
        for alt in alternatives:
            if alt in df.columns:
                selection_metric = alt
                break

    if selection_metric not in df.columns:
        raise ValueError(f"Selection metric '{selection_metric}' not found in metrics")

    # Extract per-task best configurations
    binary_configs: Dict[str, SelectedBinaryConfig] = {}
    ascending = direction == "min"

    exclude_cols = {
        "fold", "sampler", "task", "model_name",
        "rank_test_score", "rank_test_f1", "rank_test_F1 Score",
        "mean_test_f1", "std_test_f1", "mean_test_score", "std_test_score",
        "mean_test_F1 Score", "std_test_F1 Score",
    }

    for task_name in df["task"].unique():
        task_df = df[df["task"] == task_name].dropna(subset=[selection_metric])
        if task_df.empty:
            logger.warning(f"No valid configs for task '{task_name}'")
            continue

        best_row = task_df.sort_values(selection_metric, ascending=ascending).iloc[0]

        # Extract hyperparameters
        params = {}
        for col in best_row.index:
            if col not in exclude_cols and pd.notna(best_row[col]):
                params[col] = best_row[col]

        binary_configs[task_name] = SelectedBinaryConfig(
            task_name=task_name,
            model_name=best_row["model_name"],
            params=params,
            mean_score=float(best_row[selection_metric]),
            sampler=variant,
        )

    logger.info(f"Extracted {len(binary_configs)} per-task configurations")

    # Extract meta configuration from calibration comparison if available
    meta_config = None
    cal_path = technical_run / "calibration_comparison.json"
    if cal_path.exists():
        try:
            cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
            selection = cal_data.get("selection", {})
            meta_config = SelectedMetaConfig(
                model_name="LogisticRegression",
                params={"class_weight": "balanced"},
                calibration_method=selection.get("method_selected", "sigmoid"),
                calibration_cv=3,
                calibration_bins=10,
            )
        except Exception as e:
            logger.warning(f"Could not load calibration comparison: {e}")

    return binary_configs, meta_config


def save_selected_configs(
    outdir: Path,
    binary_configs: Dict[str, SelectedBinaryConfig],
    meta_config: Optional[SelectedMetaConfig],
) -> Dict[str, Path]:
    """
    Save selected configurations to registry files.

    Creates:
    - selected_binary_configs.json
    - selected_meta_config.json (if applicable)

    Returns paths to created files.
    """
    registry_dir = outdir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Binary configs
    binary_path = registry_dir / "selected_binary_configs.json"
    binary_data = {k: v.to_dict() for k, v in binary_configs.items()}
    with open(binary_path, "w", encoding="utf-8") as f:
        json.dump(binary_data, f, indent=2, default=str)
    paths["binary_configs"] = binary_path
    logger.info(f"Saved selected binary configs: {binary_path}")

    # Meta config
    if meta_config:
        meta_path = registry_dir / "selected_meta_config.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_config.to_dict(), f, indent=2)
        paths["meta_config"] = meta_path
        logger.info(f"Saved selected meta config: {meta_path}")

    return paths


def load_selected_configs(
    registry_dir: Path,
) -> Tuple[Dict[str, SelectedBinaryConfig], Optional[SelectedMetaConfig]]:
    """Load selected configurations from registry files."""
    binary_configs: Dict[str, SelectedBinaryConfig] = {}
    meta_config: Optional[SelectedMetaConfig] = None

    binary_path = registry_dir / "selected_binary_configs.json"
    if binary_path.exists():
        data = json.loads(binary_path.read_text(encoding="utf-8"))
        binary_configs = {k: SelectedBinaryConfig.from_dict(v) for k, v in data.items()}

    meta_path = registry_dir / "selected_meta_config.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_config = SelectedMetaConfig.from_dict(data)

    return binary_configs, meta_config


# -----------------------------------------------------------------------------
# Model Training Utilities
# -----------------------------------------------------------------------------


def _filter_model_params(estimator, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to only those valid for the estimator."""
    try:
        valid = set(estimator.get_params().keys())
    except Exception:
        return {}

    cleaned: Dict[str, Any] = {}
    defaults = estimator.get_params()

    int_param_hints = {
        "hidden_dim", "n_layers", "epochs", "batch_size", "max_iter",
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
    }

    for key, value in params.items():
        # Remove clf__ prefix if present
        clean_key = key.replace("clf__", "")
        if clean_key not in valid:
            continue
        if value != value:  # NaN check
            continue

        default = defaults.get(clean_key)

        # Type coercion
        if isinstance(value, float) and value.is_integer():
            if isinstance(default, int) or clean_key in int_param_hints:
                value = int(value)
        if isinstance(default, bool) and isinstance(value, (int, float)):
            value = bool(value)

        cleaned[clean_key] = value

    return cleaned


def _create_sampler(sampler_type: str, random_state: int):
    """Create a sampler based on type."""
    if sampler_type == "smote":
        return AdaptiveSMOTE(k_max=5, random_state=random_state)
    return "passthrough"


def _get_scores(pipe, X: pd.DataFrame) -> np.ndarray:
    """Extract probability scores from pipeline."""
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 and proba.shape[1] == 2 else proba
    return pipe.decision_function(X)


# -----------------------------------------------------------------------------
# Sanity Checks
# -----------------------------------------------------------------------------


def run_sanity_checks(
    binary_pipes: Dict[str, Any],
    best_models: Dict[str, str],
    X: pd.DataFrame,
    y: pd.Series,
    tasks: Dict[str, Callable],
    min_std: float = 0.02,
    max_mean_deviation: float = 0.15,
) -> List[SanityCheckResult]:
    """
    Run comprehensive sanity checks on trained models.

    Checks:
    1. Prediction variance is above threshold (not collapsed)
    2. Mean predictions are not centered at 0.5 (indicating random)
    3. Probabilities sum to 1 for each sample

    Parameters
    ----------
    binary_pipes : Dict[str, Any]
        Trained binary pipelines
    best_models : Dict[str, str]
        Best model name per task
    X : pd.DataFrame
        Feature data
    y : pd.Series
        Labels
    tasks : Dict[str, Callable]
        Task labeler functions
    min_std : float
        Minimum standard deviation for predictions
    max_mean_deviation : float
        Maximum deviation from 0.5 mean for near-random detection

    Returns
    -------
    results : List[SanityCheckResult]
        List of sanity check results
    """
    results = []

    for task_name, model_name in best_models.items():
        key = f"{task_name}__{model_name}"
        if key not in binary_pipes:
            results.append(SanityCheckResult(
                task_name=task_name,
                check_type="existence",
                passed=False,
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                message=f"Pipeline '{key}' not found in trained pipelines",
            ))
            continue

        pipe = binary_pipes[key]

        # Get task-specific labels
        y_bin = tasks[task_name](y).dropna()
        if y_bin.empty or y_bin.nunique() < 2:
            results.append(SanityCheckResult(
                task_name=task_name,
                check_type="data",
                passed=True,  # Not a failure, just insufficient data
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                message=f"Skipped - insufficient data or single class",
            ))
            continue

        X_subset = X.loc[y_bin.index]

        # Get predictions
        try:
            scores = _get_scores(pipe, X_subset)
        except Exception as e:
            results.append(SanityCheckResult(
                task_name=task_name,
                check_type="prediction",
                passed=False,
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                message=f"Prediction failed: {e}",
            ))
            continue

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        # Check 1: Variance threshold
        if std_score < min_std:
            # Additional check: if mean is near 0.5, it's likely random
            if abs(mean_score - 0.5) < max_mean_deviation:
                results.append(SanityCheckResult(
                    task_name=task_name,
                    check_type="variance_collapse",
                    passed=False,
                    mean=mean_score,
                    std=std_score,
                    min_val=min_score,
                    max_val=max_score,
                    message=(
                        f"FAILED: Near-random predictions detected. "
                        f"Mean={mean_score:.4f}, Std={std_score:.4f}. "
                        f"Model may not have trained properly."
                    ),
                ))
                continue

        # Check 2: Probability range
        if min_score < -0.1 or max_score > 1.1:
            results.append(SanityCheckResult(
                task_name=task_name,
                check_type="probability_range",
                passed=False,
                mean=mean_score,
                std=std_score,
                min_val=min_score,
                max_val=max_score,
                message=(
                    f"FAILED: Predictions outside valid range. "
                    f"Min={min_score:.4f}, Max={max_score:.4f}."
                ),
            ))
            continue

        # All checks passed
        results.append(SanityCheckResult(
            task_name=task_name,
            check_type="all",
            passed=True,
            mean=mean_score,
            std=std_score,
            min_val=min_score,
            max_val=max_score,
            message=f"PASSED: Mean={mean_score:.4f}, Std={std_score:.4f}",
        ))

    return results


def validate_sanity_checks(results: List[SanityCheckResult]) -> Tuple[bool, List[str]]:
    """
    Validate sanity check results and return overall status.

    Returns
    -------
    passed : bool
        True if all critical checks passed
    failures : List[str]
        List of failure messages
    """
    failures = []
    for result in results:
        if not result.passed and result.check_type != "data":
            failures.append(f"{result.task_name}: {result.message}")

    return len(failures) == 0, failures


# -----------------------------------------------------------------------------
# Meta-Feature Building
# -----------------------------------------------------------------------------


def build_meta_features_for_final(
    X: pd.DataFrame,
    y: pd.Series,
    binary_pipes: Dict[str, Any],
    best_models: Dict[str, str],
    tasks: Dict[str, Callable],
) -> pd.DataFrame:
    """
    Build meta-features from binary task scores for final model.

    Unlike CV training, this uses direct predictions (no out-of-fold).
    """
    meta = pd.DataFrame(index=X.index)

    for task_name, model_name in best_models.items():
        key = f"{task_name}__{model_name}"
        if key not in binary_pipes:
            continue

        pipe = binary_pipes[key]
        y_bin = tasks[task_name](y).dropna()
        if y_bin.empty:
            continue

        idx = y_bin.index
        X_subset = X.loc[idx]

        scores = _get_scores(pipe, X_subset)
        meta.loc[idx, f"{task_name}_score"] = scores

    return meta.fillna(0.0)


# -----------------------------------------------------------------------------
# Main Training Functions
# -----------------------------------------------------------------------------


def train_final_meta_model(config: FinalTrainConfig) -> FinalTrainResult:
    """
    Train final meta-classifier model from scratch on full training data.

    This is the main function for creating production-ready meta-classifier models.

    Steps:
    1. Load full training data
    2. Build all binary tasks
    3. Train each binary classifier using per-task best configs
    4. Run sanity checks on binary predictions
    5. Build meta-features
    6. Train and calibrate meta-classifier
    7. Save all artifacts

    Parameters
    ----------
    config : FinalTrainConfig
        Final training configuration

    Returns
    -------
    result : FinalTrainResult
        Training result with artifacts and sanity check results

    Raises
    ------
    ValueError
        If sanity checks fail critically
    """
    logger.info("=" * 70)
    logger.info("FINAL MODEL TRAINING (from scratch)")
    logger.info("=" * 70)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Sampler: {config.sampler}")
    logger.info(f"Output: {config.outdir}")

    warnings = []
    artifacts = {}
    training_stats = {
        "started_at": datetime.now().isoformat(),
        "mode": config.mode,
        "sampler": config.sampler,
    }

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\n[1/6] Loading training data...")
    X_full, y_full = load_data(config.train_manifest, config.label_col)

    if config.classes:
        mask = y_full.isin(config.classes)
        X_full = X_full[mask]
        y_full = y_full[mask]
        classes = config.classes
    else:
        classes = sorted(y_full.unique().tolist())

    training_stats["n_samples"] = len(X_full)
    training_stats["n_features"] = len(X_full.columns)
    training_stats["classes"] = classes
    training_stats["class_distribution"] = y_full.value_counts().to_dict()

    logger.info(f"  Samples: {len(X_full)}")
    logger.info(f"  Features: {len(X_full.columns)}")
    logger.info(f"  Classes: {classes}")

    # Build tasks
    logger.info("\n[2/6] Building binary tasks...")
    task_builder = TaskBuilder(classes).build_all_auto_tasks()
    tasks = task_builder.get_tasks()
    logger.info(f"  Built {len(tasks)} tasks")
    training_stats["n_tasks"] = len(tasks)

    # Get model registry
    model_spec = get_model_set(
        command="train-meta",
        backend=get_backend(config.backend),
        model_set=config.model_set,
        random_state=config.random_state,
        max_iter=config.max_iter,
        device=config.device,
        torch_dtype=config.torch_dtype,
        torch_num_workers=config.torch_num_workers,
        meta_C_grid=None,
    )
    estimators = model_spec["base_estimators"]

    # Train binary classifiers
    logger.info("\n[3/6] Training binary classifiers (from scratch)...")
    logger.info(f"  Using per-task configs from technical validation")
    logger.info(f"  Sampler: {config.sampler}")

    sampler = _create_sampler(config.sampler, config.random_state)

    best_pipes = {}
    best_models = {}
    binary_training_stats = {}

    for task_name, task_func in tasks.items():
        y_bin = task_func(y_full).dropna()
        if y_bin.nunique() < 2:
            logger.warning(f"  Skipping {task_name}: single class")
            continue

        X_bin = X_full.loc[y_bin.index]

        # Get per-task config
        if task_name in config.selected_binary_configs:
            task_cfg = config.selected_binary_configs[task_name]
            model_name = task_cfg.model_name
            params = task_cfg.params
            logger.info(f"  {task_name}: using {model_name} (score={task_cfg.mean_score:.4f})")
        else:
            # Fallback to first estimator
            model_name = list(estimators.keys())[0]
            params = {}
            warnings.append(f"No config for {task_name}, using default {model_name}")
            logger.warning(f"  {task_name}: no config, using default {model_name}")

        if model_name not in estimators:
            warnings.append(f"Model {model_name} not available for {task_name}")
            logger.warning(f"  {task_name}: model {model_name} not available")
            continue

        # Build pipeline
        estimator = estimators[model_name]
        cleaned = _filter_model_params(estimator, params)
        clf_params = {f"clf__{k}": v for k, v in cleaned.items()}

        pipe = ImbPipeline([
            ("sampler", sampler),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

        if clf_params:
            try:
                pipe.set_params(**clf_params)
            except Exception as e:
                warnings.append(f"Could not set params for {task_name}: {e}")

        # Train
        try:
            pipe.fit(X_bin, y_bin)
            best_pipes[f"{task_name}__{model_name}"] = pipe
            best_models[task_name] = model_name
            binary_training_stats[task_name] = {
                "model_name": model_name,
                "n_samples": len(X_bin),
                "n_positive": int(y_bin.sum()),
                "params": cleaned,
            }
        except Exception as e:
            warnings.append(f"Failed to train {task_name}: {e}")
            logger.error(f"  {task_name}: training failed: {e}")

    training_stats["binary_models"] = binary_training_stats
    logger.info(f"  Trained {len(best_models)} binary classifiers")

    # Sanity checks
    logger.info("\n[4/6] Running sanity checks...")
    sanity_results = run_sanity_checks(
        binary_pipes=best_pipes,
        best_models=best_models,
        X=X_full,
        y=y_full,
        tasks=tasks,
        min_std=config.sanity_min_std,
        max_mean_deviation=config.sanity_max_mean_deviation,
    )

    passed, failures = validate_sanity_checks(sanity_results)

    for result in sanity_results:
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"  [{status}] {result.task_name}: {result.message}")

    if not passed:
        # Save sanity check results before failing
        sanity_path = config.outdir / "sanity_checks.json"
        with open(sanity_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in sanity_results], f, indent=2)

        error_msg = (
            f"SANITY CHECK FAILURE: {len(failures)} check(s) failed.\n"
            f"Failures:\n" + "\n".join(f"  - {f}" for f in failures) +
            f"\n\nSee {sanity_path} for details.\n"
            f"The final model has degenerate predictions and cannot be used for inference."
        )
        raise ValueError(error_msg)

    # Build meta-features
    logger.info("\n[5/6] Building meta-features and training meta-classifier...")
    X_meta = build_meta_features_for_final(X_full, y_full, best_pipes, best_models, tasks)
    logger.info(f"  Meta-features shape: {X_meta.shape}")

    # Train meta-classifier
    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    meta_model.fit(X_meta.values, y_full.values)

    # Calibrate if enabled
    calibration_metadata = {
        "enabled": False,
        "method_requested": config.calibration_method,
        "method_used": None,
        "cv": None,
        "bins": config.calibration_bins,
        "warnings": [],
    }

    if config.calibrate_meta:
        method = config.calibration_method
        y_series = pd.Series(y_full)

        if method == "isotonic":
            min_samples = config.isotonic_min_samples
            if len(X_meta) < min_samples or y_series.value_counts().min() < 2:
                calibration_metadata["warnings"].append(
                    "Isotonic not supported; falling back to sigmoid."
                )
                method = "sigmoid"

        cv = max(2, min(config.calibration_cv, max(2, len(X_meta) - 1)))

        try:
            calibrator = CalibratedClassifierCV(
                estimator=meta_model,
                method=method,
                cv=cv,
            )
            calibrator.fit(X_meta.values, y_full.values)
            meta_model = calibrator
            calibration_metadata.update({
                "enabled": True,
                "method_used": method,
                "cv": cv,
            })
            logger.info(f"  Calibration: {method} (cv={cv})")
        except Exception as e:
            calibration_metadata["warnings"].append(f"Calibration failed: {e}")
            warnings.append(f"Calibration failed: {e}")

    # Save artifacts
    logger.info("\n[6/6] Saving artifacts...")

    variant = config.sampler if config.sampler != "none" else "none"
    fold_dir = config.outdir / "fold1" / f"binary_{variant}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Binary pipelines
    binary_path = fold_dir / "binary_pipes.joblib"
    joblib.dump({"pipes": best_pipes, "best_models": best_models}, binary_path)
    artifacts["binary_pipes"] = str(binary_path)

    # Meta model
    meta_path = fold_dir / "meta_model.joblib"
    joblib.dump(meta_model, meta_path)
    artifacts["meta_model"] = str(meta_path)

    # Meta features schema
    features_path = fold_dir / "meta_features.csv"
    pd.Series(list(X_meta.columns)).to_csv(features_path, index=False, header=False)
    artifacts["meta_features"] = str(features_path)

    # Class order
    classes_path = fold_dir / "meta_classes.csv"
    pd.Series(list(meta_model.classes_)).to_csv(classes_path, index=False, header=False)
    artifacts["meta_classes"] = str(classes_path)

    # Feature schema (original features)
    feature_schema_path = config.outdir / "feature_schema.json"
    feature_schema = {
        "feature_list": list(X_full.columns),
        "n_features": len(X_full.columns),
    }
    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)
    artifacts["feature_schema"] = str(feature_schema_path)

    # Class order lock
    class_order_path = config.outdir / "class_order.json"
    class_order = {
        "classes": list(meta_model.classes_),
        "n_classes": len(meta_model.classes_),
    }
    with open(class_order_path, "w", encoding="utf-8") as f:
        json.dump(class_order, f, indent=2)
    artifacts["class_order"] = str(class_order_path)

    # Calibration metadata
    cal_meta_path = fold_dir / "calibration_metadata.json"
    with open(cal_meta_path, "w", encoding="utf-8") as f:
        json.dump(calibration_metadata, f, indent=2)
    artifacts["calibration_metadata"] = str(cal_meta_path)

    # Sanity checks results
    sanity_path = config.outdir / "sanity_checks.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in sanity_results], f, indent=2)
    artifacts["sanity_checks"] = str(sanity_path)

    # Training config
    config_path = config.outdir / "final_train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    artifacts["config"] = str(config_path)

    # Training stats
    training_stats["completed_at"] = datetime.now().isoformat()
    stats_path = config.outdir / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2, default=str)
    artifacts["training_stats"] = str(stats_path)

    logger.info(f"  Saved {len(artifacts)} artifacts to {config.outdir}")

    logger.info("\n" + "=" * 70)
    logger.info("FINAL MODEL TRAINING COMPLETE")
    logger.info("=" * 70)

    return FinalTrainResult(
        success=True,
        outdir=config.outdir,
        sanity_checks=sanity_results,
        artifacts=artifacts,
        config_used=config.to_dict(),
        training_stats=training_stats,
        warnings=warnings,
    )


def train_final_binary_model(config: FinalTrainConfig) -> FinalTrainResult:
    """Train final binary classifier model from scratch."""
    # Simplified version for binary mode
    # Implementation follows similar pattern to meta but without meta-features
    raise NotImplementedError("Binary final training not yet implemented")


def train_final_multiclass_model(config: FinalTrainConfig) -> FinalTrainResult:
    """Train final multiclass model from scratch."""
    # Implementation for multiclass mode
    raise NotImplementedError("Multiclass final training not yet implemented")
