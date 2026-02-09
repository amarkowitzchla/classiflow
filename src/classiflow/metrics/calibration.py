"""Probability quality helpers (Brier score, ECE variants, calibration curves)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

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
    mode: str = "multiclass",
    binning: str = "uniform",
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Compute probability calibration metrics and curve data.

    Returns metrics dict and calibration curves keyed by curve type.
    """
    metrics: Dict[str, Any] = {
        "brier": float("nan"),
        "brier_recommended": float("nan"),
        "brier_binary": float("nan"),
        "brier_multiclass_sum": float("nan"),
        "brier_multiclass_mean": float("nan"),
        "log_loss": float("nan"),
        "ece": float("nan"),
        "ece_top1": float("nan"),
        "ece_binary_pos": float("nan"),
        "ece_ovr_macro": float("nan"),
        "mean_confidence_top1": float("nan"),
        "accuracy_top1": float("nan"),
        "confidence_gap_top1": float("nan"),
        "pred_alignment_mismatch_rate": float("nan"),
        "pred_alignment_note": "",
    }
    curves: Dict[str, pd.DataFrame] = {}
    if y_proba is None:
        return metrics, curves
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.size == 0 or y_proba.ndim != 2:
        return metrics, curves

    classes = [str(c) for c in classes]
    y_true_arr = np.array([str(v) for v in y_true])
    y_pred_arr = np.array([str(v) for v in y_pred])

    if y_proba.shape[0] != len(y_true_arr) or y_proba.shape[1] != len(classes):
        return metrics, curves
    if not np.isfinite(y_proba).all():
        return metrics, curves

    # Probability quality metrics are defined on probability outputs.
    if (y_proba < -1e-12).any() or (y_proba > 1.0 + 1e-12).any():
        return metrics, curves
    y_proba = np.clip(y_proba, 0.0, 1.0)
    row_sums = y_proba.sum(axis=1)
    if (row_sums <= 0.0).any():
        return metrics, curves
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        y_proba = y_proba / row_sums[:, None]

    if mode not in {"binary", "multiclass", "meta", "hierarchical"}:
        mode = "multiclass"
    if binning not in {"uniform", "quantile"}:
        binning = "uniform"

    # Deprecated aliases kept for one release.
    warnings.warn(
        "Metric aliases 'ece' and 'brier' are deprecated and will be removed in a future release; "
        "use 'ece_top1' and 'brier_recommended'.",
        DeprecationWarning,
        stacklevel=2,
    )

    n_classes = len(classes)
    y_pred_argmax = np.array(classes, dtype=object)[np.argmax(y_proba, axis=1)].astype(str)
    mismatch_rate = float(np.mean(y_pred_arr != y_pred_argmax)) if len(y_pred_arr) else float("nan")
    metrics["pred_alignment_mismatch_rate"] = mismatch_rate
    note = "Using argmax(y_proba) for top1 ECE by definition; y_pred mismatch indicates postprocessing."
    if mode == "hierarchical":
        note += " Mismatch may occur if hierarchical constraints alter the final predicted label."
    metrics["pred_alignment_note"] = note

    confidences = np.max(y_proba, axis=1)
    correct_top1 = (y_pred_argmax == y_true_arr).astype(float)
    if len(confidences) > 0:
        ece_top1, top1_curve = _calibration_curve(confidences, correct_top1, bins, binning=binning)
        metrics["ece_top1"] = ece_top1
        metrics["mean_confidence_top1"] = float(np.mean(confidences))
        metrics["accuracy_top1"] = float(np.mean(correct_top1))
        metrics["confidence_gap_top1"] = metrics["mean_confidence_top1"] - metrics["accuracy_top1"]
        curves["top1"] = top1_curve

    try:
        if n_classes == 2:
            positive_class = classes[1]
            y_true_binary = (y_true_arr == positive_class).astype(float)
            p_positive = y_proba[:, 1]
            metrics["brier_binary"] = float(np.mean((p_positive - y_true_binary) ** 2))
            metrics["brier_recommended"] = metrics["brier_binary"]
            ece_binary_pos, curve_binary_pos = _calibration_curve(
                p_positive,
                y_true_binary,
                bins,
                binning=binning,
            )
            metrics["ece_binary_pos"] = ece_binary_pos
            curves["binary_pos"] = curve_binary_pos
        else:
            y_bin = label_binarize(y_true_arr, classes=classes)
            if y_bin.shape[1] != y_proba.shape[1]:
                return metrics, curves
            brier_sum = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))
            metrics["brier_multiclass_sum"] = brier_sum
            metrics["brier_multiclass_mean"] = brier_sum / float(n_classes)
            metrics["brier_recommended"] = metrics["brier_multiclass_mean"]
            per_class_rows = []
            for idx, cls in enumerate(classes):
                targets = (y_true_arr == cls).astype(float)
                ece_class, curve_df = _calibration_curve(
                    y_proba[:, idx], targets, bins, binning=binning
                )
                metrics[f"ece_ovr__{cls}"] = ece_class
                per_class_rows.append({"class": cls, "ece": ece_class})
                curves[f"ovr_{cls}"] = curve_df
            if per_class_rows:
                metrics["ece_ovr_macro"] = float(np.mean([r["ece"] for r in per_class_rows]))
                curves["ovr_macro"] = pd.DataFrame(per_class_rows)
    except Exception:
        pass

    metrics["brier"] = metrics["brier_recommended"]
    metrics["ece"] = metrics["ece_top1"]

    try:
        metrics["log_loss"] = float(log_loss(y_true_arr, y_proba, labels=classes))
    except Exception:
        pass

    return metrics, curves


def _calibration_curve(
    confidences: np.ndarray,
    targets: np.ndarray,
    bins: int,
    binning: str = "uniform",
) -> Tuple[float, pd.DataFrame]:
    if bins < 1:
        bins = 10

    confidences = np.asarray(confidences, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(confidences) == 0:
        return float("nan"), pd.DataFrame()

    edges = np.linspace(0.0, 1.0, bins + 1)
    if binning == "quantile":
        quantile_edges = np.quantile(confidences, np.linspace(0.0, 1.0, bins + 1))
        quantile_edges[0] = min(0.0, float(quantile_edges[0]))
        quantile_edges[-1] = max(1.0, float(quantile_edges[-1]))
        unique_edges = np.unique(quantile_edges)
        if unique_edges.size >= 2:
            edges = unique_edges
    bin_indices = np.digitize(confidences, edges) - 1
    active_bins = len(edges) - 1
    bin_indices = np.clip(bin_indices, 0, active_bins - 1)
    records = []
    total = len(confidences)
    ece = 0.0

    for bin_id in range(active_bins):
        mask = bin_indices == bin_id
        n = int(mask.sum())
        if n == 0:
            mean_pred = 0.0
            frac_pos = 0.0
        else:
            mean_pred = float(confidences[mask].mean())
            frac_pos = float(targets[mask].mean())
        records.append(
            {
                "bin_id": bin_id,
                "n": n,
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
                "bin_start": float(edges[bin_id]),
                "bin_end": float(edges[bin_id + 1]),
            }
        )
        if total > 0:
            ece += (n / total) * abs(mean_pred - frac_pos)

    curve_df = pd.DataFrame(records)
    return float(ece), curve_df
