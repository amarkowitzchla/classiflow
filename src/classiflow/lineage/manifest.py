"""Manifest management for training and inference runs."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import classiflow

logger = logging.getLogger(__name__)


@dataclass
class TrainingRunManifest:
    """
    Manifest for a training run with full lineage and provenance.

    This provides a stable identity and traceable lineage for trained models.
    """

    # Core identity
    run_id: str  # UUID4 for this training run
    timestamp: str
    package_version: str

    # Data lineage
    training_data_path: str
    training_data_hash: str
    training_data_size_bytes: int
    training_data_row_count: Optional[int] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "binary"  # binary, meta, hierarchical

    # Environment
    python_version: str = ""
    hostname: Optional[str] = None
    git_hash: Optional[str] = None

    # Model artifacts
    artifact_registry: Dict[str, Any] = field(default_factory=dict)

    # Feature metadata
    feature_list: List[str] = field(default_factory=list)
    feature_summaries: Dict[str, Any] = field(default_factory=dict)

    # Task definitions
    task_definitions: Dict[str, Any] = field(default_factory=dict)
    best_models: Dict[str, str] = field(default_factory=dict)

    # Hierarchical metadata
    hierarchical: bool = False
    l1_classes: Optional[List[str]] = None
    l2_classes_per_branch: Optional[Dict[str, List[str]]] = None
    l3_classes_per_branch: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Handle Path objects inside config or metadata payloads.
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved training manifest to {path}")

    @classmethod
    def load(cls, path: Path) -> TrainingRunManifest:
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class InferenceRunManifest:
    """
    Manifest for an inference run with lineage to parent training run.

    This links predictions back to the model that generated them and the
    data they were generated from.
    """

    # Core identity
    inference_run_id: str  # UUID4 for this inference run
    parent_run_id: str  # Training run_id
    timestamp: str
    package_version: str

    # Data lineage
    inference_data_path: str
    inference_data_hash: str
    inference_data_size_bytes: int
    inference_data_row_count: Optional[int] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Environment
    python_version: str = ""
    hostname: Optional[str] = None

    # Model source
    model_bundle: Optional[str] = None  # Path to bundle if used
    model_run_dir: Optional[str] = None  # Path to run directory if used

    # Validation warnings
    drift_warnings: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Handle Path objects inside config or metadata payloads.
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved inference manifest to {path}")

    @classmethod
    def load(cls, path: Path) -> InferenceRunManifest:
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def create_training_manifest(
    data_path: Path,
    data_hash: str,
    data_size_bytes: int,
    data_row_count: Optional[int],
    config: Dict[str, Any],
    task_type: str = "binary",
    feature_list: Optional[List[str]] = None,
    task_definitions: Optional[Dict[str, Any]] = None,
    hierarchical: bool = False,
) -> TrainingRunManifest:
    """
    Create a training run manifest with auto-populated environment metadata.

    Parameters
    ----------
    data_path : Path
        Path to training data file
    data_hash : str
        SHA256 hash of training data
    data_size_bytes : int
        File size in bytes
    data_row_count : Optional[int]
        Number of rows in training data
    config : Dict[str, Any]
        Training configuration
    task_type : str
        Type of task (binary, meta, hierarchical)
    feature_list : Optional[List[str]]
        List of feature column names
    task_definitions : Optional[Dict[str, Any]]
        Task definitions
    hierarchical : bool
        Whether this is hierarchical

    Returns
    -------
    manifest : TrainingRunManifest
        Populated manifest
    """
    import sys

    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Get package version
    try:
        package_version = classiflow.__version__
    except AttributeError:
        package_version = "unknown"

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Hostname
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = None

    # Git hash
    git_hash = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=data_path.parent if data_path.is_file() else data_path,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    manifest = TrainingRunManifest(
        run_id=run_id,
        timestamp=timestamp,
        package_version=package_version,
        training_data_path=str(data_path),
        training_data_hash=data_hash,
        training_data_size_bytes=data_size_bytes,
        training_data_row_count=data_row_count,
        config=config,
        task_type=task_type,
        python_version=python_version,
        hostname=hostname,
        git_hash=git_hash,
        feature_list=feature_list or [],
        task_definitions=task_definitions or {},
        hierarchical=hierarchical,
    )

    logger.info(f"Created training manifest with run_id={run_id}")
    return manifest


def create_inference_manifest(
    parent_run_id: str,
    data_path: Path,
    data_hash: str,
    data_size_bytes: int,
    data_row_count: Optional[int],
    config: Dict[str, Any],
    model_source: Optional[Path] = None,
    model_is_bundle: bool = False,
) -> InferenceRunManifest:
    """
    Create an inference run manifest linked to parent training run.

    Parameters
    ----------
    parent_run_id : str
        Training run_id from parent manifest
    data_path : Path
        Path to inference data file
    data_hash : str
        SHA256 hash of inference data
    data_size_bytes : int
        File size in bytes
    data_row_count : Optional[int]
        Number of rows
    config : Dict[str, Any]
        Inference configuration
    model_source : Optional[Path]
        Path to model bundle or run directory
    model_is_bundle : bool
        Whether model_source is a bundle

    Returns
    -------
    manifest : InferenceRunManifest
        Populated manifest
    """
    import sys

    inference_run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Get package version
    try:
        package_version = classiflow.__version__
    except AttributeError:
        package_version = "unknown"

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Hostname
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = None

    manifest = InferenceRunManifest(
        inference_run_id=inference_run_id,
        parent_run_id=parent_run_id,
        timestamp=timestamp,
        package_version=package_version,
        inference_data_path=str(data_path),
        inference_data_hash=data_hash,
        inference_data_size_bytes=data_size_bytes,
        inference_data_row_count=data_row_count,
        config=config,
        python_version=python_version,
        hostname=hostname,
        model_bundle=str(model_source) if model_is_bundle else None,
        model_run_dir=str(model_source) if not model_is_bundle and model_source else None,
    )

    logger.info(f"Created inference manifest with inference_run_id={inference_run_id}, parent_run_id={parent_run_id}")
    return manifest


def load_training_manifest(run_dir: Path) -> TrainingRunManifest:
    """
    Load training manifest from run directory.

    Parameters
    ----------
    run_dir : Path
        Run directory containing run.json

    Returns
    -------
    manifest : TrainingRunManifest
        Loaded manifest
    """
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        # Try legacy run_manifest.json
        legacy_path = run_dir / "run_manifest.json"
        if legacy_path.exists():
            logger.warning(f"Using legacy manifest name: {legacy_path}")
            manifest_path = legacy_path
        else:
            raise FileNotFoundError(f"No run manifest found in {run_dir}")

    return TrainingRunManifest.load(manifest_path)


def validate_manifest_compatibility(
    training_manifest: TrainingRunManifest,
    inference_features: List[str],
) -> tuple[bool, List[str]]:
    """
    Validate that inference features are compatible with training manifest.

    Parameters
    ----------
    training_manifest : TrainingRunManifest
        Training manifest with feature_list
    inference_features : List[str]
        Feature columns from inference data

    Returns
    -------
    compatible : bool
        Whether features are compatible
    warnings : List[str]
        List of compatibility warnings
    """
    warnings = []

    training_features = set(training_manifest.feature_list)
    inference_features_set = set(inference_features)

    # Missing features
    missing = training_features - inference_features_set
    if missing:
        warnings.append(f"Missing {len(missing)} features: {sorted(list(missing))[:10]}")

    # Extra features
    extra = inference_features_set - training_features
    if extra:
        warnings.append(f"Extra {len(extra)} features (will be ignored): {sorted(list(extra))[:10]}")

    # Feature order
    if list(inference_features) != training_manifest.feature_list:
        warnings.append("Feature order differs from training (will be reordered)")

    compatible = len(missing) == 0

    return compatible, warnings
