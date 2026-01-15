"""Bundle inspection utilities."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def inspect_bundle(bundle_path: Path) -> Dict[str, Any]:
    """
    Inspect a model bundle and return metadata.

    Parameters
    ----------
    bundle_path : Path
        Path to bundle ZIP file

    Returns
    -------
    info : Dict[str, Any]
        Dictionary with bundle metadata:
        - run_manifest: training run manifest
        - artifacts: artifact registry
        - version: package version
        - readme: README content
        - file_list: list of files in bundle
        - size_mb: bundle size in MB
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"Not a valid ZIP file: {bundle_path}")

    info = {
        "bundle_path": str(bundle_path),
        "size_mb": bundle_path.stat().st_size / (1024 * 1024),
    }

    with zipfile.ZipFile(bundle_path, "r") as zf:
        # List all files
        info["file_list"] = sorted(zf.namelist())
        info["file_count"] = len(info["file_list"])

        # Load run manifest
        try:
            with zf.open("run.json") as f:
                info["run_manifest"] = json.load(f)
        except KeyError:
            logger.warning("No run.json in bundle")
            info["run_manifest"] = {}

        # Load artifact registry
        try:
            with zf.open("artifacts.json") as f:
                info["artifacts"] = json.load(f)
        except KeyError:
            logger.warning("No artifacts.json in bundle")
            info["artifacts"] = {}

        # Load version
        try:
            with zf.open("version.txt") as f:
                info["version"] = f.read().decode("utf-8").strip()
        except KeyError:
            info["version"] = "unknown"

        # Load README
        try:
            with zf.open("README.txt") as f:
                info["readme"] = f.read().decode("utf-8")
        except KeyError:
            info["readme"] = ""

        # Validate required files
        required = ["run.json", "artifacts.json", "version.txt"]
        missing = [f for f in required if f not in info["file_list"]]
        info["valid"] = len(missing) == 0
        info["missing_files"] = missing

    logger.info(f"Inspected bundle: {bundle_path} ({info['size_mb']:.2f} MB, {info['file_count']} files)")

    return info


def print_bundle_info(bundle_path: Path, verbose: bool = False) -> None:
    """
    Print human-readable bundle information.

    Parameters
    ----------
    bundle_path : Path
        Path to bundle
    verbose : bool
        Show detailed file listing
    """
    info = inspect_bundle(bundle_path)

    print("=" * 70)
    print(f"Bundle: {info['bundle_path']}")
    print("=" * 70)
    print(f"Size: {info['size_mb']:.2f} MB")
    print(f"Files: {info['file_count']}")
    print(f"Valid: {'✓' if info['valid'] else '✗'}")

    if info.get("missing_files"):
        print(f"Missing: {', '.join(info['missing_files'])}")

    print()

    # Run manifest summary
    manifest = info.get("run_manifest", {})
    if manifest:
        print("-" * 70)
        print("Training Run")
        print("-" * 70)
        print(f"Run ID: {manifest.get('run_id', 'N/A')}")
        print(f"Task Type: {manifest.get('task_type', 'N/A')}")
        print(f"Timestamp: {manifest.get('timestamp', 'N/A')}")
        print(f"Package Version: {manifest.get('package_version', 'N/A')}")
        print(f"Training Data: {manifest.get('training_data_path', 'N/A')}")
        print(f"Data Hash: {manifest.get('training_data_hash', 'N/A')[:16]}...")
        print(f"Data Rows: {manifest.get('training_data_row_count', 'N/A')}")

        if manifest.get("hierarchical"):
            print(f"Hierarchical: Yes")
            print(f"L1 Classes: {len(manifest.get('l1_classes', []))}")

        print()

    # Artifact summary
    artifacts = info.get("artifacts", {})
    if artifacts.get("folds"):
        print("-" * 70)
        print("Artifacts")
        print("-" * 70)
        for fold_name, fold_info in artifacts["folds"].items():
            print(f"{fold_name}: {fold_info.get('artifact_count', 0)} file(s)")
        print()

    # Version
    print("-" * 70)
    print(f"Version: {info.get('version', 'unknown')}")
    print("-" * 70)
    print()

    # File listing
    if verbose:
        print("-" * 70)
        print("Files")
        print("-" * 70)
        for fname in info["file_list"]:
            print(f"  {fname}")
        print()


def validate_bundle_version(bundle_path: Path, current_version: str) -> tuple[bool, List[str]]:
    """
    Validate bundle compatibility with current package version.

    Parameters
    ----------
    bundle_path : Path
        Path to bundle
    current_version : str
        Current package version

    Returns
    -------
    compatible : bool
        Whether bundle is compatible
    warnings : List[str]
        List of compatibility warnings
    """
    info = inspect_bundle(bundle_path)
    warnings = []

    # Check if bundle is valid
    if not info.get("valid"):
        warnings.append(f"Invalid bundle: missing files {info.get('missing_files', [])}")
        return False, warnings

    # Check version
    bundle_version = info.get("version", "").replace("classiflow ", "")
    if bundle_version != current_version:
        warnings.append(
            f"Version mismatch: bundle={bundle_version}, current={current_version} "
            "(may have compatibility issues)"
        )

    # Check for required manifest fields
    manifest = info.get("run_manifest", {})
    required_fields = ["run_id", "training_data_hash", "feature_list"]
    for field in required_fields:
        if field not in manifest:
            warnings.append(f"Missing manifest field: {field}")

    compatible = len([w for w in warnings if "Invalid bundle" in w or "Missing manifest" in w]) == 0

    return compatible, warnings
