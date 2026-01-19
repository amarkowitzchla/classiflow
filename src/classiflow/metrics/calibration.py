"""Probability quality helpers (Brier score, ECE, calibration curve)."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize


def compute_probability_quality(
    y_true: List,
    y_pred: List,
    y_proba: Optional[np.ndarray],
    classes: List[str],
    bins: int = 10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compute probability calibration metrics and curve data.

    Returns metrics dict and calibration curve DataFrame.
    """
    metrics: Dict[str, float] = {
        "brier": float("nan"),
        "log_loss": float("nan"),
        "ece": float("nan"),
    }
    curve_df = pd.DataFrame()
    if y_proba is None or y_proba.size == 0:
        return metrics, curve_df

    classes = [str(c) for c in classes]
    y_true_arr = np.array([str(v) for v in y_true])
    y_pred_arr = np.array([str(v) for v in y_pred])

    if y_proba.shape[1] != len(classes):
        return metrics, curve_df

    try:
        y_bin = label_binarize(y_true_arr, classes=classes)
        if y_bin.shape[1] != y_proba.shape[1]:
            return metrics, curve_df
        metrics["brier"] = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))
    except Exception:
        pass

    try:
        metrics["log_loss"] = float(log_loss(y_true_arr, y_proba, labels=classes))
    except Exception:
        pass

    confidences = np.max(y_proba, axis=1)
    correct = (y_pred_arr == y_true_arr).astype(float)
    if len(confidences) == 0:
        return metrics, curve_df

    ece, curve_df = _calibration_curve(confidences, correct, bins)
    metrics["ece"] = ece
    return metrics, curve_df


def _calibration_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> Tuple[float, pd.DataFrame]:
    if bins < 1:
        bins = 10

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(confidences, edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    records = []
    total = len(confidences)
    ece = 0.0

    for bin_id in range(bins):
        mask = bin_indices == bin_id
        n = int(mask.sum())
        if n == 0:
            mean_pred = 0.0
            frac_pos = 0.0
        else:
            mean_pred = float(confidences[mask].mean())
            frac_pos = float(correct[mask].mean())
        records.append({
            "bin_id": bin_id,
            "n": n,
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "bin_start": float(edges[bin_id]),
            "bin_end": float(edges[bin_id + 1]),
        })
        if total > 0:
            ece += (n / total) * abs(mean_pred - frac_pos)

    curve_df = pd.DataFrame(records)
    return float(ece), curve_df
