"""Artifact saving utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_nested_cv_results(results: Dict[str, Any], outdir: Path) -> None:
    """
    Save nested CV results to CSV files.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from NestedCVOrchestrator
    outdir : Path
        Output directory
    """
    logger.info(f"Saving results to {outdir}")

    # Inner CV summary
    if results.get("inner_cv_rows"):
        df = pd.DataFrame(results["inner_cv_rows"])
        df.to_csv(outdir / "metrics_inner_cv.csv", index=False)

    # Inner CV per-split
    if results.get("inner_cv_split_rows"):
        from classiflow.metrics.scorers import SCORER_ORDER
        df = pd.DataFrame(results["inner_cv_split_rows"], columns=["task_model", "fold"] + SCORER_ORDER)
        df.to_csv(outdir / "metrics_inner_cv_splits.csv", index=False)

        try:
            with pd.ExcelWriter(outdir / "metrics_inner_cv_splits.xlsx", engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="InnerCV_Splits")
                ws = writer.sheets["InnerCV_Splits"]
                for col_idx, col_name in enumerate(df.columns):
                    width = max(12, min(36, len(col_name) + 2))
                    ws.set_column(col_idx, col_idx, width)
        except Exception as e:
            logger.warning(f"Could not write Excel file: {e}")

    # Outer evaluation
    if results.get("outer_rows"):
        df = pd.DataFrame(results["outer_rows"])
        df.to_csv(outdir / "metrics_outer_binary_eval.csv", index=False)

    logger.info("Results saved")


def save_model(model: Any, path: Path, metadata: Dict[str, Any] | None = None) -> None:
    """
    Save model with optional metadata.

    Parameters
    ----------
    model : Any
        Trained model (sklearn-compatible)
    path : Path
        Output path (.joblib)
    metadata : Optional[Dict]
        Optional metadata to save alongside
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model": model}
    if metadata:
        payload["metadata"] = metadata

    joblib.dump(payload, path)
    logger.info(f"Saved model to {path}")
