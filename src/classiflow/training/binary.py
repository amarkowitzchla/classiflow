"""Binary task training with nested cross-validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from classiflow.config import TrainConfig
from classiflow.io import load_data, load_data_with_groups, validate_data
from classiflow.splitting import make_group_labels
from classiflow.training.nested_cv import NestedCVOrchestrator
from classiflow.training.probability_quality import attach_probability_quality_to_run_manifest
from classiflow.backends.registry import get_backend, get_model_set
from classiflow.artifacts import save_nested_cv_results
from classiflow.lineage.manifest import create_training_manifest
from classiflow.lineage.hashing import get_file_metadata
from classiflow.tracking import get_tracker, extract_loggable_params, summarize_metrics

logger = logging.getLogger(__name__)


def _log_torch_status(requested_device: str) -> None:
    """Log torch availability for GPU-backed binary training."""
    try:
        import torch
    except Exception as exc:
        logger.info("Torch available: no (%s)", exc)
        return

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    logger.info("Torch available: yes (cuda=%s, mps=%s)", cuda_available, mps_available)
    logger.info("Requested device: %s", requested_device)


def _torch_temperature_scaling_enabled(config: TrainConfig, backend: str) -> bool:
    enabled = str(getattr(config, "calibration_enabled", "auto")).strip().lower()
    method = str(getattr(config, "calibration_method", "sigmoid")).strip().lower()

    if method != "temperature":
        return False
    if backend != "torch":
        logger.info("Ignoring calibration_method=temperature for non-torch backend.")
        return False
    if enabled == "false":
        return False
    return True


def train_binary_task(config: TrainConfig) -> Dict[str, Any]:
    """
    Train a binary classification task with nested cross-validation.

    This is the main entry point for single binary task training.
    Loads data, validates it, runs nested CV, and saves artifacts.

    Parameters
    ----------
    config : TrainConfig
        Training configuration

    Returns
    -------
    results : Dict[str, Any]
        Training results including metrics and model paths
    """
    logger.info(f"Starting binary task training")
    data_path = config.resolved_data_path
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Label: {config.label_col}")
    logger.info(f"  Output: {config.outdir}")
    logger.info(f"  Backend: {config.backend}")

    # Initialize experiment tracker
    tracker = get_tracker(
        backend=config.tracker,
        experiment_name=config.experiment_name or "classiflow-binary",
    )
    tracker.start_run(
        run_name=config.run_name,
        tags=config.tracker_tags,
    )

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    groups = None
    if config.patient_col:
        X, y_raw, groups = load_data_with_groups(
            data_path,
            config.label_col,
            config.patient_col,
            feature_cols=config.feature_cols,
        )
    else:
        X, y_raw = load_data(
            data_path,
            config.label_col,
            feature_cols=config.feature_cols,
        )

    # Convert to binary
    if config.pos_label is None:
        # Infer minority class as positive
        vc = y_raw.value_counts()
        if len(vc) != 2:
            raise ValueError(f"Label column must be binary. Got {len(vc)} classes: {list(vc.index)}")
        pos_val = vc.idxmin()
        logger.info(f"Inferred positive class (minority): {pos_val}")
    else:
        pos_val = config.pos_label
        logger.info(f"Using explicit positive class: {pos_val}")

    y = (y_raw == pos_val).astype(int)

    if config.patient_col and groups is not None:
        patient_df = pd.DataFrame(
            {config.patient_col: groups, "label": y.values},
            index=X.index,
        )
        make_group_labels(patient_df, config.patient_col, "label")

    validate_data(X, y)

    logger.info(f"Class balance: 0={sum(y==0)}, 1={sum(y==1)}")

    # Create and save training manifest with lineage
    file_metadata = get_file_metadata(data_path)

    config_dict = config.to_dict()
    config_dict["stratification_level"] = "patient" if config.patient_col else "sample"

    manifest = create_training_manifest(
        data_path=data_path,
        data_hash=file_metadata["sha256_hash"],
        data_size_bytes=file_metadata["size_bytes"],
        data_row_count=file_metadata.get("row_count"),
        config=config_dict,
        task_type="binary",
        feature_list=X.columns.tolist(),
        task_definitions={"binary_task": f"positive_class={pos_val}"},
        hierarchical=False,
    )
    manifest.save(config.outdir / "run.json")
    logger.info(f"Saved training manifest: run_id={manifest.run_id}")

    backend = get_backend(config.backend)
    torch_temperature_scaling = _torch_temperature_scaling_enabled(config, backend)
    if backend == "sklearn" and config.device != "auto":
        logger.info("  Device setting is ignored for sklearn backend.")
    if backend == "torch":
        _log_torch_status(config.device)
        try:
            import torch
        except Exception as exc:
            raise ValueError(f"Torch backend requested but torch is unavailable: {exc}") from exc
        if config.require_torch_device:
            if config.device == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS device requested but not available; set --device cpu or fix MPS setup.")
            if config.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available; set --device cpu or fix CUDA setup.")

    model_spec = get_model_set(
        command="train-binary",
        backend=backend,
        model_set=config.model_set,
        random_state=config.random_state,
        max_iter=config.max_iter,
        device=config.device,
        torch_dtype=config.torch_dtype,
        torch_num_workers=config.torch_num_workers,
        torch_temperature_scaling=torch_temperature_scaling,
    )
    logger.info("Enabled estimators: %s", ", ".join(model_spec["estimators"].keys()))

    # Run nested CV
    orchestrator = NestedCVOrchestrator(
        outer_folds=config.outer_folds,
        inner_splits=config.inner_splits,
        inner_repeats=config.inner_repeats,
        random_state=config.random_state,
        smote_mode=config.smote_mode,
        max_iter=config.max_iter,
        calibration_bins=config.calibration_bins,
        calibration_binning=config.calibration_binning,
        estimators=model_spec["estimators"],
        param_grids=model_spec["param_grids"],
    )

    results = orchestrator.run_single_task(
        X=X,
        y=y,
        task_name="binary_task",
        outdir=config.outdir,
        groups=groups,
        patient_col=config.patient_col,
    )

    # Save results
    save_nested_cv_results(results, config.outdir)
    if isinstance(results.get("fold_probability_quality"), dict) and results["fold_probability_quality"]:
        attach_probability_quality_to_run_manifest(
            run_manifest_path=config.outdir / "run.json",
            fold_probability_quality=results["fold_probability_quality"],
        )

    # Log to experiment tracker
    tracker.log_params(extract_loggable_params(config))
    tracker.set_tags({
        "task_type": "binary",
        "backend": config.backend,
        "smote_mode": config.smote_mode,
        "run_id": manifest.run_id,
    })

    # Log summary metrics if available
    if "summary" in results:
        tracker.log_metrics(summarize_metrics(results["summary"]))

    # Log artifacts
    tracker.log_artifact(config.outdir / "run.json")
    for csv_file in config.outdir.glob("metrics_*.csv"):
        tracker.log_artifact(csv_file)
    for png_file in config.outdir.glob("*.png"):
        tracker.log_artifact(png_file)

    tracker.end_run()

    logger.info("Binary task training complete")
    return results
