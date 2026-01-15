"""Confidence metrics for inference predictions."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_confidence_metrics(
    probabilities: np.ndarray,
    predictions: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute per-sample confidence metrics from class probabilities.

    For each sample, computes:
    - confidence_max_proba: maximum probability across classes
    - confidence_margin: difference between top-1 and top-2 probabilities
    - confidence_entropy: entropy of probability distribution (multiclass only)

    Parameters
    ----------
    probabilities : np.ndarray
        Probability matrix of shape (n_samples, n_classes)
    predictions : Optional[np.ndarray]
        Predicted class indices (if not provided, uses argmax)

    Returns
    -------
    confidence_df : pd.DataFrame
        DataFrame with columns:
        - confidence_max_proba
        - confidence_margin
        - confidence_entropy
        - predicted_class_idx (if predictions not provided)
    """
    if len(probabilities.shape) == 1:
        # Binary case with single probability column
        probabilities = np.column_stack([1 - probabilities, probabilities])

    n_samples, n_classes = probabilities.shape

    # Sort probabilities in descending order
    sorted_proba = np.sort(probabilities, axis=1)[:, ::-1]

    # Max probability (top-1)
    max_proba = sorted_proba[:, 0]

    # Margin (top-1 minus top-2)
    if n_classes >= 2:
        margin = sorted_proba[:, 0] - sorted_proba[:, 1]
    else:
        margin = np.ones(n_samples)  # Binary case: margin = p - (1-p) = 2p - 1

    # Entropy
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    proba_safe = np.clip(probabilities, eps, 1.0)
    conf_entropy = entropy(proba_safe, axis=1)

    # Build DataFrame
    confidence_df = pd.DataFrame({
        "confidence_max_proba": max_proba,
        "confidence_margin": margin,
        "confidence_entropy": conf_entropy,
    })

    # Add predicted class if not provided
    if predictions is None:
        confidence_df["predicted_class_idx"] = np.argmax(probabilities, axis=1)

    return confidence_df


def assign_confidence_buckets(
    confidence_scores: pd.Series,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Assign confidence buckets based on max probability.

    Parameters
    ----------
    confidence_scores : pd.Series
        Max probability scores (0-1)
    thresholds : Optional[Dict[str, float]]
        Custom thresholds dict with keys: high_min, medium_min, low_min
        Default: high ≥0.9, medium ≥0.7, low <0.7

    Returns
    -------
    buckets : pd.Series
        Series with categorical values: "high", "medium", "low"
    """
    if thresholds is None:
        thresholds = {"high_min": 0.9, "medium_min": 0.7, "low_min": 0.0}

    buckets = pd.cut(
        confidence_scores,
        bins=[0.0, thresholds["medium_min"], thresholds["high_min"], 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    return buckets


def annotate_predictions_with_confidence(
    predictions_df: pd.DataFrame,
    probabilities: np.ndarray,
    confidence_thresholds: Optional[Dict[str, float]] = None,
    proba_col_prefix: str = "predicted_proba_",
) -> pd.DataFrame:
    """
    Annotate predictions DataFrame with confidence metrics.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame
    probabilities : np.ndarray
        Probability matrix (n_samples, n_classes)
    confidence_thresholds : Optional[Dict[str, float]]
        Custom thresholds for confidence buckets
    proba_col_prefix : str
        Prefix for probability columns in predictions_df

    Returns
    -------
    annotated_df : pd.DataFrame
        Predictions with added confidence columns
    """
    # Compute confidence metrics
    confidence_df = compute_confidence_metrics(probabilities)

    # Assign buckets
    confidence_df["confidence_bucket"] = assign_confidence_buckets(
        confidence_df["confidence_max_proba"],
        thresholds=confidence_thresholds,
    )

    # Merge with predictions
    annotated_df = pd.concat([predictions_df, confidence_df], axis=1)

    return annotated_df


def summarize_confidence_distribution(
    confidence_df: pd.DataFrame,
    by_class: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Summarize confidence metrics distribution.

    Parameters
    ----------
    confidence_df : pd.DataFrame
        DataFrame with confidence metrics
    by_class : Optional[pd.Series]
        Class labels for stratification

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics (mean, std, min, max, quartiles)
    """
    metrics = ["confidence_max_proba", "confidence_margin", "confidence_entropy"]

    if by_class is not None:
        # Stratified summary
        summary = confidence_df[metrics].groupby(by_class).describe()
    else:
        # Overall summary
        summary = confidence_df[metrics].describe()

    return summary


def filter_by_confidence(
    predictions_df: pd.DataFrame,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    confidence_col: str = "confidence_max_proba",
) -> pd.DataFrame:
    """
    Filter predictions by confidence threshold.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with confidence metrics
    min_confidence : float
        Minimum confidence threshold
    max_confidence : float
        Maximum confidence threshold
    confidence_col : str
        Column name for confidence score

    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered predictions
    """
    mask = (
        (predictions_df[confidence_col] >= min_confidence) &
        (predictions_df[confidence_col] <= max_confidence)
    )
    return predictions_df[mask].copy()


def compute_confidence_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute confidence calibration metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    calibration : Dict[str, Any]
        Dictionary with:
        - expected_calibration_error (ECE)
        - bin_edges
        - bin_accuracies
        - bin_confidences
        - bin_counts
    """
    # Get predicted probabilities for predicted class
    if len(y_proba.shape) == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])

    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute accuracy and average confidence per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = (y_pred[mask] == y_true[mask]).mean()
            bin_conf = confidences[mask].mean()
            bin_counts.append(mask.sum())
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            bin_counts.append(0)

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error (ECE)
    weights = bin_counts / bin_counts.sum()
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))

    return {
        "expected_calibration_error": float(ece),
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


def create_confidence_summary_sheet(
    predictions_df: pd.DataFrame,
    writer: pd.ExcelWriter,
    sheet_name: str = "Confidence_Summary",
) -> None:
    """
    Create Excel sheet with confidence summary.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with confidence metrics
    writer : pd.ExcelWriter
        Excel writer object
    sheet_name : str
        Sheet name
    """
    confidence_cols = [
        "confidence_max_proba",
        "confidence_margin",
        "confidence_entropy",
    ]

    # Overall summary
    summary = predictions_df[confidence_cols].describe()

    # Bucket counts
    if "confidence_bucket" in predictions_df.columns:
        bucket_counts = predictions_df["confidence_bucket"].value_counts().to_frame("count")
        bucket_pct = (bucket_counts / len(predictions_df) * 100).round(2)
        bucket_pct.columns = ["percentage"]
        bucket_summary = pd.concat([bucket_counts, bucket_pct], axis=1)
    else:
        bucket_summary = pd.DataFrame()

    # Write to sheet
    summary.to_excel(writer, sheet_name=sheet_name, startrow=0)

    if not bucket_summary.empty:
        bucket_summary.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=len(summary) + 3,
        )

    logger.info(f"Created confidence summary sheet: {sheet_name}")
