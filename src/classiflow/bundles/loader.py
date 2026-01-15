"""Bundle loading utilities for inference."""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

import pandas as pd
import numpy as np

from classiflow.lineage import TrainingRunManifest

logger = logging.getLogger(__name__)


class BundleLoader:
    """
    Loader for model bundles that handles extraction and artifact loading.

    This provides a unified interface for loading models from bundles,
    abstracting away the ZIP extraction and file management.
    """

    def __init__(self, bundle_path: Path, extract_dir: Optional[Path] = None):
        """
        Initialize bundle loader.

        Parameters
        ----------
        bundle_path : Path
            Path to bundle ZIP file
        extract_dir : Optional[Path]
            Directory to extract bundle to; if None, use temporary directory
        """
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.bundle_path}")

        self._temp_dir = None
        self.extract_dir = extract_dir

        # Extract bundle
        if extract_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="classiflow_bundle_")
            self.extract_dir = Path(self._temp_dir)
        else:
            self.extract_dir = Path(extract_dir)
            self.extract_dir.mkdir(parents=True, exist_ok=True)

        self._extract_bundle()
        self._load_metadata()

    def _extract_bundle(self) -> None:
        """Extract bundle contents."""
        logger.info(f"Extracting bundle to {self.extract_dir}")
        with zipfile.ZipFile(self.bundle_path, "r") as zf:
            zf.extractall(self.extract_dir)

    def _load_metadata(self) -> None:
        """Load bundle metadata."""
        # Try to load run manifest - could be new or legacy format
        run_json_path = self.extract_dir / "run.json"
        run_manifest_path = self.extract_dir / "run_manifest.json"

        manifest_path = None
        if run_json_path.exists():
            manifest_path = run_json_path
        elif run_manifest_path.exists():
            manifest_path = run_manifest_path
        else:
            raise FileNotFoundError(f"No run.json or run_manifest.json in bundle")

        # Load the JSON data
        with open(manifest_path, "r") as f:
            data = json.load(f)

        # Check if it's new format (has run_id) or legacy (has config at root)
        if "run_id" in data and "training_data_hash" in data:
            # New format - load directly
            self.manifest = TrainingRunManifest(**data)
            logger.info(f"Loaded manifest: run_id={self.manifest.run_id}")
        else:
            # Legacy format - convert to new structure
            logger.warning(f"Bundle uses legacy manifest format ({manifest_path.name})")

            # Detect task type from legacy data
            config = data.get("config", {})
            if "label_l1" in config:
                task_type = "hierarchical"
            elif "classes" in config or "tasks_json" in config:
                task_type = "meta"
            else:
                task_type = "binary"

            # Create minimal manifest from legacy data
            self.manifest = TrainingRunManifest(
                run_id=data.get("run_id", "unknown"),
                timestamp=data.get("timestamp", "unknown"),
                package_version=data.get("package_version", "0.1.0"),
                training_data_path=config.get("data_csv", "unknown"),
                training_data_hash="0" * 64,  # Placeholder
                training_data_size_bytes=0,
                training_data_row_count=None,
                config=config,
                task_type=task_type,
                feature_list=data.get("feature_list", []) or [],
            )
            logger.info(f"Loaded legacy manifest: task_type={self.manifest.task_type}")

        # Load artifact registry
        artifacts_path = self.extract_dir / "artifacts.json"
        if artifacts_path.exists():
            with open(artifacts_path, "r") as f:
                self.artifacts = json.load(f)
        else:
            logger.warning("No artifacts.json in bundle")
            self.artifacts = {}

    def get_fold_dir(self, fold: int = 1) -> Path:
        """
        Get path to fold directory.

        Parameters
        ----------
        fold : int
            Fold number (1-indexed)

        Returns
        -------
        fold_dir : Path
            Path to fold directory
        """
        fold_dir = self.extract_dir / f"fold{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold {fold} not found in bundle")
        return fold_dir

    def list_folds(self) -> list[int]:
        """
        List available folds in bundle.

        Returns
        -------
        folds : List[int]
            List of fold numbers
        """
        folds = []
        for item in self.extract_dir.iterdir():
            if item.is_dir() and item.name.startswith("fold"):
                try:
                    fold_num = int(item.name.replace("fold", ""))
                    folds.append(fold_num)
                except ValueError:
                    pass
        return sorted(folds)

    def load_artifact(self, artifact_path: str) -> Any:
        """
        Load an artifact from bundle.

        Parameters
        ----------
        artifact_path : str
            Relative path to artifact (e.g., "fold1/binary_smote/binary_pipes.joblib")

        Returns
        -------
        artifact : Any
            Loaded artifact
        """
        import joblib

        full_path = self.extract_dir / artifact_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        return joblib.load(full_path)

    def cleanup(self) -> None:
        """Clean up temporary extraction directory."""
        if self._temp_dir is not None:
            logger.debug(f"Cleaning up temporary directory: {self._temp_dir}")
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def load_bundle(
    bundle_path: Path,
    fold: int = 1,
    extract_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a model bundle for inference.

    This is a convenience function that extracts the bundle and returns
    the necessary components for inference.

    Parameters
    ----------
    bundle_path : Path
        Path to bundle ZIP file
    fold : int
        Fold number to load (1-indexed)
    extract_dir : Optional[Path]
        Directory to extract to; if None, use temporary directory

    Returns
    -------
    bundle_data : Dict[str, Any]
        Dictionary containing:
        - loader: BundleLoader instance
        - manifest: TrainingRunManifest
        - fold_dir: Path to fold directory
        - artifacts: artifact registry
    """
    loader = BundleLoader(bundle_path, extract_dir=extract_dir)

    return {
        "loader": loader,
        "manifest": loader.manifest,
        "fold_dir": loader.get_fold_dir(fold),
        "artifacts": loader.artifacts,
        "available_folds": loader.list_folds(),
    }
