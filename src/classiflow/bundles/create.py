"""Bundle creation utilities."""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import shutil

import classiflow

logger = logging.getLogger(__name__)


def create_bundle(
    run_dir: Path,
    out_bundle: Path,
    fold: Optional[int] = None,
    include_all_folds: bool = False,
    include_metrics: bool = True,
    description: Optional[str] = None,
) -> Path:
    """
    Create a portable model bundle from a training run directory.

    A bundle is a self-contained ZIP archive that includes:
    - run.json (training manifest with run_id, data hash, etc.)
    - artifacts.json (artifact registry)
    - version.txt (package version)
    - README.txt (bundle metadata and usage)
    - fold{N}/ directories with serialized models
    - Optional: metrics CSVs

    Parameters
    ----------
    run_dir : Path
        Training run directory
    out_bundle : Path
        Output bundle path (will add .zip if missing)
    fold : Optional[int]
        Specific fold to include (1-indexed); if None and not include_all_folds, include fold 1
    include_all_folds : bool
        Include all folds in bundle
    include_metrics : bool
        Include metrics CSVs
    description : Optional[str]
        Optional description for README

    Returns
    -------
    bundle_path : Path
        Path to created bundle
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Ensure .zip extension
    out_bundle = Path(out_bundle)
    if out_bundle.suffix != ".zip":
        out_bundle = out_bundle.with_suffix(".zip")

    out_bundle.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating bundle from {run_dir} -> {out_bundle}")

    # Validate required files
    run_manifest_path = run_dir / "run.json"
    if not run_manifest_path.exists():
        # Try legacy name
        run_manifest_path = run_dir / "run_manifest.json"
        if not run_manifest_path.exists():
            raise FileNotFoundError(f"No run.json or run_manifest.json in {run_dir}")

    # Load manifest for metadata
    with open(run_manifest_path, "r") as f:
        manifest = json.load(f)

    # Determine which folds to include
    if include_all_folds:
        fold_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("fold")])
    else:
        fold_num = fold if fold is not None else 1
        fold_dir = run_dir / f"fold{fold_num}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        fold_dirs = [fold_dir]

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {run_dir}")

    logger.info(f"Including {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}")

    # Create bundle
    with zipfile.ZipFile(out_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Core metadata files
        zf.write(run_manifest_path, arcname="run.json")
        logger.debug("Added run.json")

        training_config_path = run_dir / "training_config.json"
        if training_config_path.exists():
            zf.write(training_config_path, arcname="training_config.json")
            logger.debug("Added training_config.json")

        inference_config_path = run_dir / "inference_config.json"
        if inference_config_path.exists():
            zf.write(inference_config_path, arcname="inference_config.json")
            logger.debug("Added inference_config.json")

        # Version file
        version_txt = f"classiflow {classiflow.__version__}\n"
        zf.writestr("version.txt", version_txt)
        logger.debug("Added version.txt")

        # Create artifact registry
        artifact_registry = _create_artifact_registry(run_dir, fold_dirs, manifest)
        zf.writestr("artifacts.json", json.dumps(artifact_registry, indent=2))
        logger.debug("Added artifacts.json")

        # README
        readme_content = _create_readme(manifest, artifact_registry, description)
        zf.writestr("README.txt", readme_content)
        logger.debug("Added README.txt")

        # Add fold directories
        for fold_dir in fold_dirs:
            _add_fold_to_bundle(zf, fold_dir, run_dir)

        # Add metrics if requested
        if include_metrics:
            _add_metrics_to_bundle(zf, run_dir)

    bundle_size_mb = out_bundle.stat().st_size / (1024 * 1024)
    logger.info(f"Bundle created: {out_bundle} ({bundle_size_mb:.2f} MB)")

    return out_bundle


def _create_artifact_registry(
    run_dir: Path,
    fold_dirs: List[Path],
    manifest: dict,
) -> dict:
    """Create artifact registry for bundle."""
    registry = {
        "bundle_created": datetime.now().isoformat(),
        "source_run_dir": str(run_dir),
        "run_id": manifest.get("run_id", "unknown"),
        "task_type": manifest.get("task_type", "unknown"),
        "folds": {},
    }

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        fold_artifacts = []

        # Scan fold directory
        for item in fold_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(run_dir)
                fold_artifacts.append({
                    "path": str(rel_path),
                    "size_bytes": item.stat().st_size,
                    "suffix": item.suffix,
                })

        registry["folds"][fold_name] = {
            "artifact_count": len(fold_artifacts),
            "artifacts": fold_artifacts,
        }

    return registry


def _add_fold_to_bundle(zf: zipfile.ZipFile, fold_dir: Path, run_dir: Path) -> None:
    """Add fold directory contents to bundle."""
    for item in fold_dir.rglob("*"):
        if item.is_file():
            arcname = str(item.relative_to(run_dir))
            zf.write(item, arcname=arcname)
            logger.debug(f"Added {arcname}")


def _add_metrics_to_bundle(zf: zipfile.ZipFile, run_dir: Path) -> None:
    """Add metrics CSV files to bundle."""
    metric_patterns = [
        "metrics_*.csv",
        "metrics_*.xlsx",
        "fold_averages*.csv",
        "summary*.csv",
    ]

    added_count = 0
    for pattern in metric_patterns:
        for metric_file in run_dir.glob(pattern):
            if metric_file.is_file():
                arcname = metric_file.name
                zf.write(metric_file, arcname=arcname)
                logger.debug(f"Added metric: {arcname}")
                added_count += 1

    logger.info(f"Added {added_count} metric file(s)")


def _create_readme(manifest: dict, registry: dict, description: Optional[str]) -> str:
    """Create README content for bundle."""
    lines = [
        "=" * 70,
        "classiflow Model Bundle",
        "=" * 70,
        "",
        f"Bundle Created: {registry['bundle_created']}",
        f"Package Version: {manifest.get('package_version', 'unknown')}",
        f"Run ID: {registry['run_id']}",
        f"Task Type: {registry['task_type']}",
        "",
        "-" * 70,
        "Contents",
        "-" * 70,
        "",
        "Core Files:",
        "  - run.json:       Training manifest with run_id, data lineage, config",
        "  - training_config.json: Training config (hierarchical runs)",
        "  - inference_config.json: Inference config (optional)",
        "  - artifacts.json: Registry of model artifacts in this bundle",
        "  - version.txt:    Package version used for training",
        "  - README.txt:     This file",
        "",
        "Model Artifacts:",
    ]

    for fold_name, fold_info in registry["folds"].items():
        lines.append(f"  - {fold_name}/:  {fold_info['artifact_count']} artifact(s)")

    lines.extend([
        "",
        "-" * 70,
        "Usage",
        "-" * 70,
        "",
        "Command-line inference:",
        "  classiflow infer --bundle model_bundle.zip --data-csv test.csv --outdir results/",
        "",
        "Python API:",
        "  from classiflow.bundles import load_bundle",
        "  bundle = load_bundle('model_bundle.zip')",
        "  predictions = bundle.predict(X)",
        "",
        "-" * 70,
        "Training Information",
        "-" * 70,
        "",
        f"Training Data: {manifest.get('training_data_path', 'N/A')}",
        f"Data Hash: {manifest.get('training_data_hash', 'N/A')}",
        f"Data Rows: {manifest.get('training_data_row_count', 'N/A')}",
        f"Timestamp: {manifest.get('timestamp', 'N/A')}",
        "",
    ])

    if description:
        lines.extend([
            "-" * 70,
            "Description",
            "-" * 70,
            "",
            description,
            "",
        ])

    lines.extend([
        "-" * 70,
        "For more information:",
        "  https://github.com/alexmarkowitz/classiflow",
        "=" * 70,
    ])

    return "\n".join(lines)
