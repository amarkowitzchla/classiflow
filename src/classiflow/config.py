"""Configuration dataclasses for training pipelines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def _resolve_data_path(
    data: Optional[Path] = None,
    data_csv: Optional[Path] = None,
) -> Path:
    """
    Resolve data path from --data or --data-csv options.

    Parameters
    ----------
    data : Path, optional
        New --data option (preferred)
    data_csv : Path, optional
        Legacy --data-csv option (deprecated)

    Returns
    -------
    Path
        Resolved data path

    Raises
    ------
    ValueError
        If neither option is provided
    """
    if data is not None:
        return Path(data)
    elif data_csv is not None:
        logger.warning(
            "--data-csv is deprecated; use --data instead. "
            "Support for --data-csv will be removed in a future version."
        )
        return Path(data_csv)
    else:
        raise ValueError("Either --data or --data-csv must be provided")


@dataclass
class TrainConfig:
    """Configuration for binary task training."""

    # Data - support both new --data and legacy --data-csv
    data_csv: Optional[Path] = None  # Legacy, deprecated
    data_path: Optional[Path] = None  # New, preferred
    label_col: str = ""
    pos_label: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    patient_col: Optional[str] = None  # Column with patient/slide IDs for stratification (optional)
    group_col: Optional[str] = None  # Deprecated alias for patient/group stratification

    # Output
    outdir: Path = Path("derived")

    # Cross-validation
    outer_folds: int = 3
    inner_splits: int = 5
    inner_repeats: int = 2
    random_state: int = 42

    # SMOTE
    smote_mode: Literal["off", "on", "both"] = "off"
    smote_k_neighbors: int = 5

    # Models
    max_iter: int = 10000
    backend: Literal["sklearn", "torch"] = "sklearn"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    model_set: Optional[str] = None
    torch_num_workers: int = 0
    torch_dtype: Literal["float32", "float16"] = "float32"
    require_torch_device: bool = False

    # Experiment tracking (optional)
    tracker: Optional[Literal["mlflow", "wandb"]] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracker_tags: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Convert string paths to Path objects and resolve data path."""
        # Resolve data path (prefer data_path over data_csv)
        if self.data_path is not None:
            self.data_path = Path(self.data_path)
            # Also set data_csv for backward compatibility with existing code
            if self.data_csv is None:
                self.data_csv = self.data_path
        elif self.data_csv is not None:
            self.data_csv = Path(self.data_csv)
            # Set data_path from data_csv for forward compatibility
            self.data_path = self.data_csv
        # Note: We allow both to be None during construction, validation happens at use time

        self.outdir = Path(self.outdir)

    @property
    def resolved_data_path(self) -> Path:
        """Get the resolved data path (prefers data_path over data_csv)."""
        if self.data_path is not None:
            return self.data_path
        if self.data_csv is not None:
            return self.data_csv
        raise ValueError("No data path configured. Set data_path or data_csv.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict with Path objects as strings."""
        d = asdict(self)
        d["data_csv"] = str(self.data_csv) if self.data_csv else None
        d["data_path"] = str(self.data_path) if self.data_path else None
        d["outdir"] = str(self.outdir)
        return d

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class MetaConfig(TrainConfig):
    """Configuration for meta-classifier training (multiclass)."""

    # Task definition
    classes: Optional[List[str]] = None
    tasks_json: Optional[Path] = None
    tasks_only: bool = False  # If True and tasks_json provided, skip auto OvR/pairwise tasks

    # Meta-classifier
    meta_C_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 1, 10])
    calibrate_meta: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    calibration_cv: int = 3
    calibration_bins: int = 10
    calibration_isotonic_min_samples: int = 100

    def __post_init__(self):
        """Convert string paths to Path objects."""
        super().__post_init__()
        if self.tasks_json is not None:
            self.tasks_json = Path(self.tasks_json)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict with Path objects as strings."""
        d = super().to_dict()
        if self.tasks_json is not None:
            d["tasks_json"] = str(self.tasks_json)
        d.update(
            calibrate_meta=self.calibrate_meta,
            calibration_method=self.calibration_method,
            calibration_cv=self.calibration_cv,
            calibration_bins=self.calibration_bins,
            calibration_isotonic_min_samples=self.calibration_isotonic_min_samples,
        )
        return d


@dataclass
class MulticlassConfig(TrainConfig):
    """Configuration for direct multiclass training."""

    classes: Optional[List[str]] = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    estimator_mode: Literal["all", "torch_only", "cpu_only"] = "all"
    group_stratify: bool = True
    logreg_solver: str = "saga"
    logreg_multi_class: str = "auto"
    logreg_penalty: str = "l2"  # deprecated in sklearn 1.8; retained for compatibility
    logreg_max_iter: int = 5000
    logreg_tol: float = 1e-3
    logreg_C: float = 1.0
    logreg_class_weight: Optional[str] = "balanced"
    logreg_n_jobs: int = -1

    def __post_init__(self):
        """Convert string paths to Path objects."""
        super().__post_init__()


@dataclass
class HierarchicalConfig:
    """
    Configuration for hierarchical nested-CV training with optional patient-level stratification.

    Supports:
    - Single-label classification (binary or multiclass)
    - Hierarchical two-level classification (L1 → branch-specific L2)
    - Patient-level stratified splits (no data leakage) when patient_col is provided
    - Sample-level stratified splits when patient_col is None
    - PyTorch MLP with CUDA/MPS support
    - Early stopping and hyperparameter tuning
    """

    # Data - support both new --data and legacy --data-csv
    data_csv: Optional[Path] = None  # Legacy, deprecated
    data_path: Optional[Path] = None  # New, preferred
    patient_col: Optional[str] = None  # Column with patient/slide IDs for stratification (optional)
    label_l1: str = ""  # Level-1 label column (required)
    label_l2: Optional[str] = None  # Level-2 label column (optional, enables hierarchical mode)
    l2_classes: Optional[List[str]] = None  # Subset of L2 classes to include
    min_l2_classes_per_branch: int = 2  # Minimum L2 classes required to train a branch
    feature_cols: Optional[List[str]] = None  # Explicit feature columns (auto-detect if None)

    # Output
    outdir: Path = Path("derived_hierarchical")
    output_format: Literal["xlsx", "csv"] = "xlsx"

    # Cross-validation
    outer_folds: int = 3
    inner_splits: int = 3
    random_state: int = 42

    # PyTorch MLP
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mlp_epochs: int = 100
    mlp_batch_size: int = 256
    mlp_hidden: int = 128  # Base hidden dimension
    mlp_dropout: float = 0.3
    early_stopping_patience: int = 10

    # SMOTE
    use_smote: bool = False
    smote_k_neighbors: int = 5

    # Logging
    verbose: int = 1  # 0=minimal, 1=standard, 2=detailed

    # Experiment tracking (optional)
    tracker: Optional[Literal["mlflow", "wandb"]] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracker_tags: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Convert string paths to Path objects and resolve data path."""
        # Resolve data path (prefer data_path over data_csv)
        if self.data_path is not None:
            self.data_path = Path(self.data_path)
            # Also set data_csv for backward compatibility with existing code
            if self.data_csv is None:
                self.data_csv = self.data_path
        elif self.data_csv is not None:
            self.data_csv = Path(self.data_csv)
            # Set data_path from data_csv for forward compatibility
            self.data_path = self.data_csv

        self.outdir = Path(self.outdir)

    @property
    def resolved_data_path(self) -> Path:
        """Get the resolved data path (prefers data_path over data_csv)."""
        if self.data_path is not None:
            return self.data_path
        if self.data_csv is not None:
            return self.data_csv
        raise ValueError("No data path configured. Set data_path or data_csv.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict with Path objects as strings."""
        d = asdict(self)
        d["data_csv"] = str(self.data_csv) if self.data_csv else None
        d["data_path"] = str(self.data_path) if self.data_path else None
        d["outdir"] = str(self.outdir)
        return d

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def hierarchical(self) -> bool:
        """Whether hierarchical mode is enabled."""
        return self.label_l2 is not None


@dataclass
class RunManifest:
    """Manifest for a training run with reproducibility metadata."""

    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_hash: Optional[str] = None
    python_version: str = field(default_factory=lambda: f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}")
    hostname: Optional[str] = None

    def __post_init__(self):
        """Try to capture git hash and hostname."""
        if self.git_hash is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    self.git_hash = result.stdout.strip()
            except Exception:
                pass

        if self.hostname is None:
            try:
                import socket
                self.hostname = socket.gethostname()
            except Exception:
                pass

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_config(cls, config: TrainConfig | MetaConfig | HierarchicalConfig) -> RunManifest:
        """Create manifest from a config object."""
        return cls(config=config.to_dict())
