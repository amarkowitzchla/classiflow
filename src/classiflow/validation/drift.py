"""Feature drift detection and validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_summary(
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-feature distribution summary statistics.

    For each feature, computes:
    - mean
    - std (standard deviation)
    - median
    - q25, q75 (25th and 75th percentiles for IQR)
    - missing_rate (fraction of NaN/inf values)
    - dtype

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    feature_names : Optional[List[str]]
        Feature names (default: X.columns)

    Returns
    -------
    summary : Dict[str, Dict[str, float]]
        Feature summaries {feature_name: {stat: value}}
    """
    if feature_names is None:
        feature_names = X.columns.tolist()

    summary = {}

    for col in feature_names:
        if col not in X.columns:
            logger.warning(f"Feature '{col}' not in data, skipping")
            continue

        vals = X[col].values

        # Handle non-numeric gracefully
        try:
            vals_clean = vals[np.isfinite(vals)]

            if len(vals_clean) > 0:
                summary[col] = {
                    "mean": float(np.mean(vals_clean)),
                    "std": float(np.std(vals_clean)),
                    "median": float(np.median(vals_clean)),
                    "q25": float(np.percentile(vals_clean, 25)),
                    "q75": float(np.percentile(vals_clean, 75)),
                    "missing_rate": float(1.0 - len(vals_clean) / len(vals)),
                    "count_clean": int(len(vals_clean)),
                    "count_total": int(len(vals)),
                }
            else:
                # All missing
                summary[col] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "missing_rate": 1.0,
                    "count_clean": 0,
                    "count_total": int(len(vals)),
                }
        except Exception as e:
            logger.warning(f"Could not compute summary for '{col}': {e}")
            summary[col] = {
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "missing_rate": 1.0,
                "count_clean": 0,
                "count_total": int(len(vals)),
            }

    return summary


def compute_drift_scores(
    train_summary: Dict[str, Dict[str, float]],
    inference_summary: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compute drift scores between training and inference feature distributions.

    For each feature, computes:
    - z_shift: (mean_inf - mean_train) / std_train
    - missing_delta: missing_rate_inf - missing_rate_train
    - median_shift: (median_inf - median_train) / IQR_train

    Parameters
    ----------
    train_summary : Dict[str, Dict[str, float]]
        Training feature summaries
    inference_summary : Dict[str, Dict[str, float]]
        Inference feature summaries

    Returns
    -------
    drift_df : pd.DataFrame
        DataFrame with columns: feature, z_shift, missing_delta, median_shift
    """
    records = []

    for feature in train_summary.keys():
        if feature not in inference_summary:
            logger.warning(f"Feature '{feature}' not in inference data")
            continue

        train_stats = train_summary[feature]
        inf_stats = inference_summary[feature]

        # Z-shift for means
        if train_stats["std"] > 0:
            z_shift = (inf_stats["mean"] - train_stats["mean"]) / train_stats["std"]
        else:
            z_shift = 0.0

        # Missing delta
        missing_delta = inf_stats["missing_rate"] - train_stats["missing_rate"]

        # Median shift (normalized by IQR)
        iqr_train = train_stats["q75"] - train_stats["q25"]
        if iqr_train > 0:
            median_shift = (inf_stats["median"] - train_stats["median"]) / iqr_train
        else:
            median_shift = 0.0

        records.append({
            "feature": feature,
            "z_shift": z_shift,
            "missing_delta": missing_delta,
            "median_shift": median_shift,
            "mean_train": train_stats["mean"],
            "mean_inference": inf_stats["mean"],
            "std_train": train_stats["std"],
            "std_inference": inf_stats["std"],
            "missing_rate_train": train_stats["missing_rate"],
            "missing_rate_inference": inf_stats["missing_rate"],
        })

    drift_df = pd.DataFrame(records)

    # Add absolute scores for ranking
    if not drift_df.empty:
        drift_df["abs_z_shift"] = drift_df["z_shift"].abs()
        drift_df["abs_missing_delta"] = drift_df["missing_delta"].abs()
        drift_df["abs_median_shift"] = drift_df["median_shift"].abs()

    return drift_df


def detect_drift(
    drift_df: pd.DataFrame,
    z_threshold: float = 3.0,
    missing_threshold: float = 0.1,
    median_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detect features with significant drift.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Drift scores from compute_drift_scores
    z_threshold : float
        Threshold for z_shift (e.g., 3.0 = 3 standard deviations)
    missing_threshold : float
        Threshold for missing_delta (e.g., 0.1 = 10% difference)
    median_threshold : float
        Threshold for median_shift (normalized by IQR)

    Returns
    -------
    flagged_features : pd.DataFrame
        Features exceeding thresholds
    warnings : List[str]
        Human-readable warnings
    """
    if drift_df.empty:
        return drift_df, []

    flagged = drift_df[
        (drift_df["abs_z_shift"] > z_threshold) |
        (drift_df["abs_missing_delta"] > missing_threshold) |
        (drift_df["abs_median_shift"] > median_threshold)
    ].copy()

    # Sort by severity (max absolute drift)
    flagged["max_drift"] = flagged[["abs_z_shift", "abs_missing_delta", "abs_median_shift"]].max(axis=1)
    flagged = flagged.sort_values("max_drift", ascending=False)

    # Generate warnings
    warnings = []
    if not flagged.empty:
        warnings.append(f"⚠️ {len(flagged)} feature(s) show significant drift:")
        for idx, row in flagged.head(10).iterrows():
            reasons = []
            if row["abs_z_shift"] > z_threshold:
                reasons.append(f"z-shift={row['z_shift']:.2f}")
            if row["abs_missing_delta"] > missing_threshold:
                reasons.append(f"missing Δ={row['missing_delta']:.2%}")
            if row["abs_median_shift"] > median_threshold:
                reasons.append(f"median-shift={row['median_shift']:.2f}")

            warnings.append(f"  • {row['feature']}: {', '.join(reasons)}")

        if len(flagged) > 10:
            warnings.append(f"  ... and {len(flagged) - 10} more")

    return flagged, warnings


def create_drift_report(
    drift_df: pd.DataFrame,
    flagged_features: pd.DataFrame,
    output_dir: Path,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Path]:
    """
    Create drift report files (CSV and Excel).

    Parameters
    ----------
    drift_df : pd.DataFrame
        Full drift scores
    flagged_features : pd.DataFrame
        Features exceeding thresholds
    output_dir : Path
        Output directory
    thresholds : Optional[Dict[str, float]]
        Thresholds used for detection

    Returns
    -------
    output_files : Dict[str, Path]
        Paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # CSV
    csv_path = output_dir / "feature_drift_summary.csv"
    drift_df.to_csv(csv_path, index=False)
    output_files["csv"] = csv_path
    logger.info(f"Saved drift summary CSV: {csv_path}")

    # Excel with multiple sheets
    xlsx_path = output_dir / "feature_drift_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        # All features
        drift_df.to_excel(writer, sheet_name="Drift", index=False)

        # Flagged features
        if not flagged_features.empty:
            flagged_features.to_excel(writer, sheet_name="Drift_Flagged", index=False)

        # Thresholds sheet
        if thresholds:
            thresh_df = pd.DataFrame([thresholds]).T
            thresh_df.columns = ["threshold"]
            thresh_df.to_excel(writer, sheet_name="Thresholds")

        # Format
        workbook = writer.book
        format_header = workbook.add_format({"bold": True, "bg_color": "#D9E1F2"})

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col_num, col_name in enumerate(drift_df.columns):
                worksheet.write(0, col_num, col_name, format_header)

    output_files["xlsx"] = xlsx_path
    logger.info(f"Saved drift summary Excel: {xlsx_path}")

    return output_files


def save_feature_summaries(
    summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Save feature summaries to JSON.

    Parameters
    ----------
    summary : Dict[str, Dict[str, float]]
        Feature summaries
    output_path : Path
        Output path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved feature summaries: {output_path}")


def load_feature_summaries(input_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load feature summaries from JSON.

    Parameters
    ----------
    input_path : Path
        Input path

    Returns
    -------
    summary : Dict[str, Dict[str, float]]
        Feature summaries
    """
    with open(input_path, "r") as f:
        summary = json.load(f)
    logger.info(f"Loaded feature summaries: {input_path}")
    return summary


def create_drift_banner_message(
    flagged_count: int,
    total_features: int,
    top_drifted: Optional[List[str]] = None,
) -> str:
    """
    Create a concise drift warning banner message for UI.

    Parameters
    ----------
    flagged_count : int
        Number of flagged features
    total_features : int
        Total number of features
    top_drifted : Optional[List[str]]
        Top drifted feature names

    Returns
    -------
    message : str
        Warning banner text
    """
    if flagged_count == 0:
        return "✓ No significant feature drift detected"

    pct = (flagged_count / total_features * 100) if total_features > 0 else 0

    message = f"⚠️ Feature Drift Warning: {flagged_count}/{total_features} features ({pct:.1f}%) show significant drift"

    if top_drifted:
        message += f"\nTop drifted: {', '.join(top_drifted[:5])}"

    return message
