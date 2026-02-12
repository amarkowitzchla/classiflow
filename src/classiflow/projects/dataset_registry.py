"""Dataset registration, hashing, and schema inference."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from classiflow.data import load_table
from classiflow.lineage.hashing import get_file_metadata
from classiflow.projects.project_models import (
    DatasetEntry,
    DatasetRegistry,
    DatasetSchema,
    DatasetStats,
    ProjectConfig,
)

logger = logging.getLogger(__name__)


def _infer_feature_representation(columns: pd.Index) -> Dict[str, Optional[str]]:
    col_lc = {col.lower(): col for col in columns}
    for candidate in ("feature_path", "features_path", "embedding_path"):
        if candidate in col_lc:
            return {"representation": "feature_path", "feature_path_column": col_lc[candidate]}
    return {"representation": "wide", "feature_path_column": None}


def _infer_schema(df: pd.DataFrame, config: ProjectConfig) -> DatasetSchema:
    columns = list(df.columns)
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    rep_info = _infer_feature_representation(df.columns)

    key_cols = {
        config.key_columns.label,
    }
    if config.key_columns.sample_id:
        key_cols.add(config.key_columns.sample_id)
    if config.key_columns.patient_id:
        key_cols.add(config.key_columns.patient_id)
    if config.key_columns.slide_id:
        key_cols.add(config.key_columns.slide_id)
    if config.key_columns.specimen_id:
        key_cols.add(config.key_columns.specimen_id)

    feature_cols = []
    if rep_info["representation"] == "wide":
        for col in df.columns:
            if col in key_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

    return DatasetSchema(
        columns=columns,
        dtypes=dtypes,
        feature_representation=rep_info["representation"],
        feature_columns=feature_cols,
        feature_path_column=rep_info["feature_path_column"],
    )


def _summarize_stats(df: pd.DataFrame, config: ProjectConfig) -> DatasetStats:
    label_col = config.key_columns.label
    labels = {}
    if label_col in df.columns:
        labels = df[label_col].astype(str).value_counts().to_dict()
    patients = None
    if config.key_columns.patient_id and config.key_columns.patient_id in df.columns:
        patients = int(df[config.key_columns.patient_id].nunique())
    return DatasetStats(rows=len(df), patients=patients, labels=labels)


def register_dataset(
    registry_path: Path,
    config: ProjectConfig,
    dataset_type: str,
    manifest_path: Path,
    git_hash: Optional[str] = None,
) -> DatasetEntry:
    """Register a dataset manifest and persist to registry."""
    dataset_type = dataset_type.lower()
    if dataset_type not in {"train", "test"}:
        raise ValueError("dataset_type must be 'train' or 'test'")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = load_table(manifest_path)
    required_cols = {config.key_columns.label}
    if config.key_columns.sample_id:
        required_cols.add(config.key_columns.sample_id)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    schema = _infer_schema(df, config)
    stats = _summarize_stats(df, config)

    metadata = get_file_metadata(manifest_path)
    sha256 = metadata["sha256_hash"]
    size_bytes = int(metadata["size_bytes"])
    registry = DatasetRegistry.load(registry_path)

    previous_hashes = []
    dirty = False
    existing = registry.datasets.get(dataset_type)
    if existing and existing.sha256 != sha256:
        dirty = True
        previous_hashes = list(existing.previous_hashes)
        previous_hashes.append(existing.sha256)

    entry = DatasetEntry(
        dataset_type=dataset_type,
        manifest_path=str(manifest_path),
        sha256=sha256,
        size_bytes=size_bytes,
        registered_at=datetime.utcnow().isoformat(),
        git_hash=git_hash,
        data_schema=schema,
        stats=stats,
        dirty=dirty,
        previous_hashes=previous_hashes,
    )

    registry.datasets[dataset_type] = entry
    registry.save(registry_path)
    logger.info("Registered %s dataset: %s", dataset_type, manifest_path)
    return entry


def verify_manifest_hash(manifest_path: Path, expected_hash: str) -> bool:
    """Verify manifest hash without parsing rows."""
    actual = compute_file_hash(manifest_path)
    return actual == expected_hash
