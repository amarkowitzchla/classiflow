"""Adapters for parsing run manifests and metrics files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunManifestNormalized:
    """Normalized run manifest combining data from multiple sources."""

    # Identity
    project_id: str
    phase: str
    run_id: str  # Short hex ID (directory name)
    run_key: str  # Composite: project_id:phase:run_id

    # Timestamps
    created_at: Optional[datetime] = None

    # Training info
    task_type: Optional[str] = None  # binary, meta, hierarchical
    training_data_path: Optional[str] = None
    training_data_hash: Optional[str] = None
    training_data_row_count: Optional[int] = None

    # Config
    config: dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None

    # Lineage
    lineage: dict[str, Any] = field(default_factory=dict)
    lineage_path: Optional[str] = None
    classiflow_version: Optional[str] = None
    git_hash: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    command: Optional[str] = None

    # Features
    feature_list: list[str] = field(default_factory=list)
    feature_count: int = 0

    # Metrics paths discovered
    metrics_summary_paths: list[str] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)

    # Full UUID from run.json (if present)
    manifest_run_id: Optional[str] = None

    # Source tracking
    source_files: list[str] = field(default_factory=list)


def parse_run_manifest(run_dir: Path, project_id: str, phase: str) -> RunManifestNormalized:
    """
    Parse run manifest from run.json (or legacy run_manifest.json).

    Falls back to inferring from lineage.json + metrics files if no manifest.

    Parameters
    ----------
    run_dir : Path
        Run directory path
    project_id : str
        Project identifier
    phase : str
        Phase name

    Returns
    -------
    RunManifestNormalized
        Normalized manifest data
    """
    run_id = run_dir.name
    run_key = f"{project_id}:{phase}:{run_id}"

    manifest = RunManifestNormalized(
        project_id=project_id,
        phase=phase,
        run_id=run_id,
        run_key=run_key,
    )

    # Try run.json first
    run_json = run_dir / "run.json"
    if run_json.exists():
        manifest = _parse_run_json(run_json, manifest)
        manifest.source_files.append("run.json")
    else:
        # Try legacy name
        legacy = run_dir / "run_manifest.json"
        if legacy.exists():
            manifest = _parse_run_json(legacy, manifest)
            manifest.source_files.append("run_manifest.json")

    # Always try to load lineage.json for additional info
    lineage_path = run_dir / "lineage.json"
    if lineage_path.exists():
        manifest = _merge_lineage(lineage_path, manifest)
        manifest.source_files.append("lineage.json")

    # Discover metrics and report paths
    manifest = _discover_metrics_paths(run_dir, manifest)
    manifest = _discover_report_paths(run_dir, manifest)

    # Check for resolved config
    config_resolved = run_dir / "config.resolved.yaml"
    if config_resolved.exists():
        manifest.config_path = str(config_resolved)

    return manifest


def _parse_run_json(path: Path, manifest: RunManifestNormalized) -> RunManifestNormalized:
    """Parse run.json and merge into manifest."""
    try:
        with open(path) as f:
            data = json.load(f)

        # Core fields
        manifest.manifest_run_id = data.get("run_id")
        manifest.task_type = data.get("task_type")

        # Timestamp
        ts = data.get("timestamp")
        if ts:
            try:
                manifest.created_at = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass

        # Data info
        manifest.training_data_path = data.get("training_data_path")
        manifest.training_data_hash = data.get("training_data_hash")
        manifest.training_data_row_count = data.get("training_data_row_count")

        # Config
        manifest.config = data.get("config", {})

        # Environment
        manifest.classiflow_version = data.get("package_version")
        manifest.git_hash = data.get("git_hash")
        manifest.python_version = data.get("python_version")

        # Features
        features = data.get("feature_list", [])
        manifest.feature_list = features
        manifest.feature_count = len(features)

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse {path}: {e}")

    return manifest


def _merge_lineage(path: Path, manifest: RunManifestNormalized) -> RunManifestNormalized:
    """Merge lineage.json data into manifest."""
    try:
        with open(path) as f:
            data = json.load(f)

        manifest.lineage = data
        manifest.lineage_path = str(path)

        # Phase from lineage (uppercase)
        lineage_phase = data.get("phase", "").lower()
        if lineage_phase and not manifest.phase:
            manifest.phase = lineage_phase

        # Timestamps - prefer lineage if not set
        if not manifest.created_at:
            ts = data.get("timestamp_local") or data.get("timestamp_utc")
            if ts:
                try:
                    # Handle timezone-aware timestamps
                    if "+" in ts or ts.endswith("Z"):
                        manifest.created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        manifest.created_at = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass

        # Version info - fill if not set
        if not manifest.classiflow_version:
            manifest.classiflow_version = data.get("classiflow_version")
        if not manifest.git_hash:
            manifest.git_hash = data.get("git_hash")
        if not manifest.python_version:
            manifest.python_version = data.get("python_version")

        manifest.platform = data.get("platform")
        manifest.command = data.get("command")

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse lineage {path}: {e}")

    return manifest


def _discover_metrics_paths(run_dir: Path, manifest: RunManifestNormalized) -> RunManifestNormalized:
    """Discover metrics file paths."""
    paths = []

    # Check for summary files at root
    for name in ["metrics_summary.json", "metrics.json"]:
        p = run_dir / name
        if p.exists():
            paths.append(str(p.relative_to(run_dir)))

    # Check for CSV/XLSX metrics at root
    for suffix in [".csv", ".xlsx"]:
        for p in run_dir.glob(f"metrics*{suffix}"):
            paths.append(str(p.relative_to(run_dir)))

    # Check metrics/ subdirectory
    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        for p in metrics_dir.glob("*.csv"):
            paths.append(str(p.relative_to(run_dir)))

    manifest.metrics_summary_paths = sorted(set(paths))
    return manifest


def _discover_report_paths(run_dir: Path, manifest: RunManifestNormalized) -> RunManifestNormalized:
    """Discover report file paths."""
    paths = []

    # Check reports/ subdirectory
    reports_dir = run_dir / "reports"
    if reports_dir.is_dir():
        for p in reports_dir.glob("*.md"):
            paths.append(str(p.relative_to(run_dir)))
        for p in reports_dir.glob("*.html"):
            paths.append(str(p.relative_to(run_dir)))

    manifest.report_paths = sorted(set(paths))
    return manifest


def parse_lineage(path: Path) -> dict[str, Any]:
    """Parse lineage.json file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse lineage {path}: {e}")
        return {}


def _parse_outer_cv_metrics_csv(run_dir: Path) -> dict[str, Any]:
    """
    Parse outer CV metrics from CSV (technical_validation phase).

    Returns summary (mean of validation folds) and per_fold data.
    """
    import csv

    result: dict[str, Any] = {"summary": {}, "per_fold": {}}

    # Look for outer CV metrics CSV
    csv_candidates = [
        run_dir / "metrics_outer_meta_eval.csv",
        run_dir / "metrics_outer_multiclass_eval.csv",
        run_dir / "metrics_outer_cv.csv",
        run_dir / "outer_cv_metrics.csv",
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if not csv_path:
        return result

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Filter to validation folds only (phase=val)
        val_rows = [r for r in rows if r.get("phase") == "val"]
        if not val_rows:
            # If no phase column, assume all rows are validation
            val_rows = rows

        if not val_rows:
            return result

        # Collect metrics per fold
        metric_columns = [
            "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
            "roc_auc_ovr_macro", "mcc"
        ]

        per_fold: dict[str, list[float]] = {}
        for col in metric_columns:
            values = []
            for row in val_rows:
                if col in row and row[col]:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, TypeError):
                        pass
            if values:
                per_fold[col] = values

        # Compute summary as mean of folds
        summary = {}
        for col, values in per_fold.items():
            if values:
                summary[col] = sum(values) / len(values)

        result["summary"] = summary
        result["per_fold"] = per_fold

    except (OSError, csv.Error) as e:
        logger.warning(f"Failed to parse outer CV metrics CSV: {e}")

    return result


def parse_metrics(run_dir: Path, phase: str) -> dict[str, Any]:
    """
    Parse metrics from various formats based on phase.

    Parameters
    ----------
    run_dir : Path
        Run directory
    phase : str
        Phase name for context

    Returns
    -------
    dict
        Parsed metrics with 'summary', 'per_fold', 'per_class', etc.
    """
    result: dict[str, Any] = {
        "summary": {},
        "per_fold": {},
        "per_class": [],
        "confusion_matrix": None,
        "roc_auc": None,
    }

    # For technical_validation, try CSV first (more complete)
    if phase == "technical_validation":
        csv_result = _parse_outer_cv_metrics_csv(run_dir)
        if csv_result["summary"]:
            result["summary"] = csv_result["summary"]
            result["per_fold"] = csv_result["per_fold"]

    # Try metrics_summary.json (technical_validation)
    summary_path = run_dir / "metrics_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                data = json.load(f)
            # Merge with existing (don't overwrite CSV data)
            for key, value in data.get("summary", {}).items():
                if key not in result["summary"]:
                    result["summary"][key] = value
            for key, value in data.get("per_fold", {}).items():
                if key not in result["per_fold"]:
                    result["per_fold"][key] = value
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse metrics_summary.json: {e}")

    # Try metrics.json (independent_test)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                data = json.load(f)

            overall = data.get("overall", data)

            # Extract summary metrics
            for key in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "log_loss"]:
                if key in overall:
                    result["summary"][key] = overall[key]

            # Per-class metrics
            if "per_class" in overall:
                result["per_class"] = overall["per_class"]

            # Confusion matrix
            if "confusion_matrix" in overall:
                result["confusion_matrix"] = overall["confusion_matrix"]

            # ROC AUC
            if "roc_auc" in overall:
                result["roc_auc"] = overall["roc_auc"]

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse metrics.json: {e}")

    return result


def infer_manifest_from_directory(run_dir: Path, project_id: str, phase: str) -> RunManifestNormalized:
    """
    Infer minimal manifest when no explicit manifest file exists.

    Uses directory metadata and known file patterns.
    """
    run_id = run_dir.name
    run_key = f"{project_id}:{phase}:{run_id}"

    manifest = RunManifestNormalized(
        project_id=project_id,
        phase=phase,
        run_id=run_id,
        run_key=run_key,
        source_files=["inferred"],
    )

    # Try to get timestamp from directory or files
    try:
        stat = run_dir.stat()
        manifest.created_at = datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        pass

    # Look for any JSON files to infer task type
    for json_file in run_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "task_type" in data:
                manifest.task_type = data["task_type"]
                break
        except (json.JSONDecodeError, OSError):
            continue

    # Discover metrics and report paths
    manifest = _discover_metrics_paths(run_dir, manifest)
    manifest = _discover_report_paths(run_dir, manifest)

    return manifest
