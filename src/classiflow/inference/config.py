"""Configuration dataclasses for inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
import json


@dataclass
class InferenceConfig:
    """
    Configuration for inference pipeline.

    Parameters
    ----------
    run_dir : Path
        Directory containing trained model artifacts
    data_path : Optional[Path]
        Input data file (.csv, .parquet) or directory (parquet dataset)
    data_csv : Optional[Path]
        [DEPRECATED] Input CSV file with features for inference. Use data_path instead.
    output_dir : Path
        Directory for inference outputs
    id_col : Optional[str]
        Column name for sample ID (default: None)
    label_col : Optional[str]
        Ground-truth label column for evaluation (default: None)
    strict_features : bool
        If True, fail if required features are missing; if False, fill with zeros/median
    lenient_fill_strategy : Literal["zero", "median"]
        Strategy for filling missing features in lenient mode
    max_roc_curves : int
        Maximum number of per-class ROC curves to generate
    hierarchical_output : bool
        Include hierarchical level outputs (L1, L2, L3) if applicable
    device : str
        Device for PyTorch models: "auto", "cpu", "cuda", or "mps"
    batch_size : int
        Batch size for inference
    include_plots : bool
        Generate ROC/AUC/confusion matrix plots
    include_excel : bool
        Generate Excel metrics workbook
    verbose : int
        Verbosity level (0=minimal, 1=standard, 2=detailed)
    """

    # Required
    run_dir: Path
    output_dir: Path

    # Data - support both new --data and legacy --data-csv
    data_path: Optional[Path] = None  # New, preferred
    data_csv: Optional[Path] = None  # Legacy, deprecated

    # Optional data handling
    id_col: Optional[str] = None
    label_col: Optional[str] = None

    # Feature alignment
    strict_features: bool = True
    lenient_fill_strategy: Literal["zero", "median"] = "zero"

    # Output control
    max_roc_curves: int = 10
    hierarchical_output: bool = True
    include_plots: bool = True
    include_excel: bool = True

    # Compute settings
    device: str = "auto"
    batch_size: int = 512

    # Logging
    verbose: int = 1

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.run_dir = Path(self.run_dir)
        self.output_dir = Path(self.output_dir)

        # Convert data paths
        if self.data_path is not None:
            self.data_path = Path(self.data_path)
        if self.data_csv is not None:
            self.data_csv = Path(self.data_csv)

        # Cross-set for backward compatibility: if only data_csv provided, also set data_path
        if self.data_path is None and self.data_csv is not None:
            self.data_path = self.data_csv
        elif self.data_csv is None and self.data_path is not None:
            self.data_csv = self.data_path

        # Validate paths
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        data_path = self.resolved_data_path
        if data_path.is_file() and not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        elif data_path.is_dir() and not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        elif not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

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
        return {
            "run_dir": str(self.run_dir),
            "data_path": str(self.data_path) if self.data_path else None,
            "data_csv": str(self.data_csv) if self.data_csv else None,
            "output_dir": str(self.output_dir),
            "id_col": self.id_col,
            "label_col": self.label_col,
            "strict_features": self.strict_features,
            "lenient_fill_strategy": self.lenient_fill_strategy,
            "max_roc_curves": self.max_roc_curves,
            "hierarchical_output": self.hierarchical_output,
            "include_plots": self.include_plots,
            "include_excel": self.include_excel,
            "device": self.device,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
        }

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class RunManifest:
    """
    Manifest describing a trained run's artifacts and configuration.

    This is created during training and loaded during inference to ensure
    compatibility and reproducibility.
    """

    # Training metadata
    training_config: Dict[str, Any]
    timestamp: str
    package_version: Optional[str] = None
    git_hash: Optional[str] = None

    # Data schema
    label_col: str = None
    feature_list: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)

    # Task definitions
    task_type: Literal["binary", "meta", "hierarchical", "multiclass"] = "binary"
    task_definitions: Dict[str, Any] = field(default_factory=dict)
    best_models: Dict[str, str] = field(default_factory=dict)  # task -> model_name

    # Model artifacts
    fold_count: int = 1
    hierarchical: bool = False
    l1_classes: Optional[List[str]] = None
    l2_classes_per_branch: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_config": self.training_config,
            "timestamp": self.timestamp,
            "package_version": self.package_version,
            "git_hash": self.git_hash,
            "label_col": self.label_col,
            "feature_list": self.feature_list,
            "preprocessing_steps": self.preprocessing_steps,
            "task_type": self.task_type,
            "task_definitions": self.task_definitions,
            "best_models": self.best_models,
            "fold_count": self.fold_count,
            "hierarchical": self.hierarchical,
            "l1_classes": self.l1_classes,
            "l2_classes_per_branch": self.l2_classes_per_branch,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
