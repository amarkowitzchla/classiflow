"""Binary task training with nested cross-validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from classiflow.config import TrainConfig
from classiflow.io import load_data, validate_data
from classiflow.training.nested_cv import NestedCVOrchestrator
from classiflow.artifacts import save_nested_cv_results
from classiflow.lineage.manifest import create_training_manifest
from classiflow.lineage.hashing import get_file_metadata

logger = logging.getLogger(__name__)


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

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # Load and validate data
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

    validate_data(X, y)

    logger.info(f"Class balance: 0={sum(y==0)}, 1={sum(y==1)}")

    # Create and save training manifest with lineage
    file_metadata = get_file_metadata(data_path)

    manifest = create_training_manifest(
        data_path=data_path,
        data_hash=file_metadata["sha256_hash"],
        data_size_bytes=file_metadata["size_bytes"],
        data_row_count=file_metadata.get("row_count"),
        config=config.to_dict(),
        task_type="binary",
        feature_list=X.columns.tolist(),
        task_definitions={"binary_task": f"positive_class={pos_val}"},
        hierarchical=False,
    )
    manifest.save(config.outdir / "run.json")
    logger.info(f"Saved training manifest: run_id={manifest.run_id}")

    # Run nested CV
    orchestrator = NestedCVOrchestrator(
        outer_folds=config.outer_folds,
        inner_splits=config.inner_splits,
        inner_repeats=config.inner_repeats,
        random_state=config.random_state,
        smote_mode=config.smote_mode,
        max_iter=config.max_iter,
    )

    results = orchestrator.run_single_task(
        X=X,
        y=y,
        task_name="binary_task",
        outdir=config.outdir,
    )

    # Save results
    save_nested_cv_results(results, config.outdir)

    logger.info("Binary task training complete")
    return results
