"""Configuration dataclasses for training pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
import json
from datetime import datetime


@dataclass
class TrainConfig:
    """Configuration for binary task training."""

    # Data
    data_csv: Path
    label_col: str
    pos_label: Optional[str] = None
    feature_cols: Optional[List[str]] = None

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

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_csv = Path(self.data_csv)
        self.outdir = Path(self.outdir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict with Path objects as strings."""
        d = asdict(self)
        d["data_csv"] = str(self.data_csv)
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
        return d


@dataclass
class MulticlassConfig(TrainConfig):
    """Configuration for direct multiclass training."""

    classes: Optional[List[str]] = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

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

    # Data
    data_csv: Path
    patient_col: Optional[str] = None  # Column with patient/slide IDs for stratification (optional)
    label_l1: str = ...  # Level-1 label column (required)
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

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_csv = Path(self.data_csv)
        self.outdir = Path(self.outdir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict with Path objects as strings."""
        d = asdict(self)
        d["data_csv"] = str(self.data_csv)
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
