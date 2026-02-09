"""Shared probability-quality artifact helpers for training runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def serialize_probability_quality_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Convert non-finite metric values to JSON-safe values."""
    serialized: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if value is None:
            serialized[key] = None
            continue
        try:
            value_float = float(value)
            if value_float != value_float:
                serialized[key] = None
            else:
                serialized[key] = value_float
        except Exception:
            serialized[key] = value
    return serialized


def write_probability_quality_curve_artifacts(
    *,
    var_dir: Path,
    final_variant: str,
    final_curves: Dict[str, pd.DataFrame],
    uncal_curves: Optional[Dict[str, pd.DataFrame]] = None,
    cal_curves: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Write calibration curve CSVs with compatibility and variant-tagged outputs."""
    var_dir.mkdir(parents=True, exist_ok=True)
    uncal_curves = uncal_curves or {}
    cal_curves = cal_curves or {}

    top1_curve = final_curves.get("top1")
    if top1_curve is not None and not top1_curve.empty:
        top1_curve.to_csv(var_dir / "calibration_curve.csv", index=False)
        top1_curve.to_csv(var_dir / "calibration_curve_top1.csv", index=False)
        top1_curve.to_csv(var_dir / f"calibration_curve_top1_{final_variant}.csv", index=False)

    for curve_name, curve_df in final_curves.items():
        if curve_df is None or curve_df.empty:
            continue
        curve_df.to_csv(var_dir / f"calibration_curve_{curve_name}.csv", index=False)
        curve_df.to_csv(var_dir / f"calibration_curve_{curve_name}_{final_variant}.csv", index=False)

    for variant_name, curves in (("uncalibrated", uncal_curves), ("calibrated", cal_curves)):
        for curve_name, curve_df in curves.items():
            if curve_df is None or curve_df.empty:
                continue
            curve_df.to_csv(var_dir / f"calibration_curve_{curve_name}_{variant_name}.csv", index=False)


def attach_probability_quality_to_run_manifest(
    *,
    run_manifest_path: Path,
    fold_probability_quality: Dict[str, Any],
) -> None:
    """Store fold-level probability-quality payload in run.json artifact_registry."""
    if not run_manifest_path.exists():
        return
    try:
        payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load run manifest for probability-quality update: %s", exc)
        return

    artifact_registry = payload.get("artifact_registry")
    if not isinstance(artifact_registry, dict):
        artifact_registry = {}
        payload["artifact_registry"] = artifact_registry

    final_variants = [
        (entry or {}).get("final_variant")
        for entry in fold_probability_quality.values()
        if isinstance(entry, dict)
    ]
    unique_final = sorted({v for v in final_variants if v})
    final_variant = unique_final[0] if len(unique_final) == 1 else "mixed"

    artifact_registry["probability_quality"] = {
        "final_variant": final_variant,
        "folds": fold_probability_quality,
    }
    try:
        run_manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist run manifest probability-quality payload: %s", exc)
