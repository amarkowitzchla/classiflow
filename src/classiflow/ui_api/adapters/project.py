"""Adapters for parsing project configuration and registry files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfigNormalized:
    """Normalized project configuration."""

    # Identity
    id: str  # Directory name
    name: str  # Human-readable name
    description: Optional[str] = None
    owner: Optional[str] = None

    # Task info
    task_mode: Optional[str] = None  # binary, meta, hierarchical
    label_column: Optional[str] = None

    # Data paths
    train_manifest: Optional[str] = None
    test_manifest: Optional[str] = None

    # Validation settings
    outer_folds: int = 3
    inner_folds: int = 5
    seed: int = 42

    # Model info
    candidates: list[str] = field(default_factory=list)
    primary_metrics: list[str] = field(default_factory=list)

    # Source
    config_path: Optional[str] = None


@dataclass
class DatasetInfo:
    """Dataset registry entry."""

    dataset_type: str  # train or test
    manifest_path: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    registered_at: Optional[datetime] = None
    columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    dirty: bool = False


@dataclass
class ThresholdsConfig:
    """Promotion thresholds configuration."""

    technical_validation: dict[str, Any] = field(default_factory=dict)
    independent_test: dict[str, Any] = field(default_factory=dict)
    promotion_logic: str = "ALL_REQUIRED_AND_STABILITY"
    promotion: dict[str, Any] = field(default_factory=dict)
    allow_override: bool = True
    require_comment: bool = True
    require_approver: bool = True


@dataclass
class DecisionResult:
    """Promotion gate decision."""

    decision: str  # PASS, FAIL
    timestamp: Optional[datetime] = None
    technical_run: Optional[str] = None
    test_run: Optional[str] = None
    reasons: list[str] = field(default_factory=list)
    override_enabled: bool = False
    override_comment: Optional[str] = None
    override_approver: Optional[str] = None


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    """Safely load YAML file, returning empty dict on error."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError) as e:
        logger.warning(f"Failed to load YAML {path}: {e}")
        return {}


def parse_project_config(project_dir: Path) -> ProjectConfigNormalized:
    """
    Parse project.yaml and return normalized config.

    Also infers ID from directory name if project.yaml is missing.
    """
    project_id = project_dir.name
    config_path = project_dir / "project.yaml"

    # Default values from directory name
    # Format: PROJECT_ID__slugified_name
    parts = project_id.split("__", 1)
    short_id = parts[0] if parts else project_id
    name = parts[1].replace("_", " ").title() if len(parts) > 1 else short_id

    config = ProjectConfigNormalized(
        id=project_id,
        name=name,
    )

    if not config_path.exists():
        return config

    data = _safe_load_yaml(config_path)
    if not data:
        return config

    config.config_path = str(config_path)

    # Project info
    project_section = data.get("project", {})
    if project_section.get("name"):
        config.name = project_section["name"]
    config.description = project_section.get("description")
    config.owner = project_section.get("owner")

    # Task info
    task_section = data.get("task", {})
    config.task_mode = task_section.get("mode")

    # Key columns
    key_cols = data.get("key_columns", {})
    config.label_column = key_cols.get("label")

    # Data paths
    data_section = data.get("data", {})
    if data_section.get("train"):
        config.train_manifest = data_section["train"].get("manifest")
    if data_section.get("test"):
        config.test_manifest = data_section["test"].get("manifest")

    # Validation
    validation = data.get("validation", {})
    nested_cv = validation.get("nested_cv", {})
    config.outer_folds = nested_cv.get("outer_folds", 3)
    config.inner_folds = nested_cv.get("inner_folds", 5)
    config.seed = nested_cv.get("seed", 42)

    # Models
    models = data.get("models", {})
    config.candidates = models.get("candidates", [])

    # Metrics
    metrics = data.get("metrics", {})
    config.primary_metrics = metrics.get("primary", [])

    return config


def parse_datasets_registry(project_dir: Path) -> dict[str, DatasetInfo]:
    """Parse registry/datasets.yaml."""
    registry_path = project_dir / "registry" / "datasets.yaml"
    if not registry_path.exists():
        return {}

    data = _safe_load_yaml(registry_path)
    datasets_data = data.get("datasets", {})

    result = {}
    for key, entry in datasets_data.items():
        if not isinstance(entry, dict):
            continue

        info = DatasetInfo(dataset_type=entry.get("dataset_type", key))
        info.manifest_path = entry.get("manifest_path")
        info.sha256 = entry.get("sha256")
        info.size_bytes = entry.get("size_bytes")

        # Registered timestamp
        reg_at = entry.get("registered_at")
        if reg_at:
            try:
                info.registered_at = datetime.fromisoformat(reg_at)
            except (ValueError, TypeError):
                pass

        # Stats
        stats = entry.get("stats", {})
        info.row_count = stats.get("rows")
        info.label_distribution = stats.get("labels", {})

        # Schema
        schema = entry.get("schema", {})
        info.columns = schema.get("columns", [])
        info.feature_columns = schema.get("feature_columns", [])

        info.dirty = entry.get("dirty", False)

        result[key] = info

    return result


def parse_thresholds(project_dir: Path) -> ThresholdsConfig:
    """Parse registry/thresholds.yaml."""
    path = project_dir / "registry" / "thresholds.yaml"
    if not path.exists():
        return ThresholdsConfig()

    data = _safe_load_yaml(path)
    if not data:
        return ThresholdsConfig()

    config = ThresholdsConfig()

    # Phase thresholds
    config.technical_validation = data.get("technical_validation", {})
    config.independent_test = data.get("independent_test", {})
    config.promotion_logic = data.get("promotion_logic", "ALL_REQUIRED_AND_STABILITY")
    config.promotion = data.get("promotion", {})

    # Override settings
    override = data.get("override", {})
    config.allow_override = override.get("allow_override", True)
    config.require_comment = override.get("require_comment", True)
    config.require_approver = override.get("require_approver", True)

    return config


def parse_decision(project_dir: Path) -> Optional[DecisionResult]:
    """Parse promotion/decision.yaml."""
    path = project_dir / "promotion" / "decision.yaml"
    if not path.exists():
        return None

    data = _safe_load_yaml(path)
    if not data:
        return None

    result = DecisionResult(decision=data.get("decision", "PENDING"))

    # Timestamp
    ts = data.get("timestamp")
    if ts:
        try:
            result.timestamp = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass

    result.technical_run = data.get("technical_run")
    result.test_run = data.get("test_run")
    result.reasons = data.get("reasons", [])

    # Override
    override = data.get("override", {})
    result.override_enabled = override.get("enabled", False)
    result.override_comment = override.get("comment")
    result.override_approver = override.get("approver")

    return result


def get_project_updated_at(project_dir: Path) -> Optional[datetime]:
    """Get the most recent modification time from key project files."""
    candidates = [
        project_dir / "project.yaml",
        project_dir / "promotion" / "decision.yaml",
        project_dir / "registry" / "datasets.yaml",
    ]

    latest = None
    for path in candidates:
        if path.exists():
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if latest is None or mtime > latest:
                    latest = mtime
            except OSError:
                continue

    # Also check runs directory for most recent run
    runs_dir = project_dir / "runs"
    if runs_dir.is_dir():
        for phase_dir in runs_dir.iterdir():
            if phase_dir.is_dir():
                for run_dir in phase_dir.iterdir():
                    if run_dir.is_dir():
                        try:
                            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                            if latest is None or mtime > latest:
                                latest = mtime
                        except OSError:
                            continue

    return latest
