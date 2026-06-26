"""Metrics computation for inference evaluation."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from classiflow.metrics.decision import compute_decision_metrics
logger = logging.getLogger(__name__)


def _as_str_array(values: np.ndarray) -> np.ndarray:
    """Normalize labels for sklearn metrics so numeric/string class names align."""
    return np.array([str(v) for v in values], dtype=object)


def _ordered_union(*label_groups: List[str]) -> List[str]:
    """Return labels in first-seen order without duplicates."""
    labels: List[str] = []
    seen = set()
    for group in label_groups:
        for label in group:
            label = str(label)
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : Optional[np.ndarray]
        Predicted probabilities (n_samples, n_classes)
    class_names : Optional[List[str]]
        Class names

    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary containing:
        - Overall metrics (accuracy, balanced_accuracy, f1_macro, etc.)
        - Per-class metrics (precision, recall, f1, support)
        - Confusion matrix
        - ROC AUC (if probabilities provided)
    """
    metrics = {}

    # Filter to valid samples (remove NaN labels)
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = _as_str_array(np.array(y_true)[valid_mask])
    y_pred_clean = _as_str_array(np.array(y_pred)[valid_mask])

    if len(y_true_clean) == 0:
        return {"error": "No valid samples for evaluation"}

    n_samples = len(y_true_clean)
    metrics["n_samples"] = n_samples
    metrics_warnings: List[str] = []

    def _add_warning(message: str) -> None:
        if message not in metrics_warnings:
            metrics_warnings.append(message)

    # Overall metrics
    metrics["accuracy"] = float(accuracy_score(y_true_clean, y_pred_clean))
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true_clean, y_pred_clean)
        )
    for caught_warning in caught_warnings:
        warning_message = str(caught_warning.message)
        if "y_pred contains classes not in y_true" in warning_message:
            warning = (
                "Predicted labels include classes absent from the test labels; "
                "balanced accuracy was computed with zero support for those classes."
            )
            logger.warning(warning)
            _add_warning(warning)
        else:
            warnings.warn(caught_warning.message, category=caught_warning.category)

    # Macro/weighted/micro F1
    for avg in ["macro", "weighted", "micro"]:
        metrics[f"f1_{avg}"] = float(f1_score(y_true_clean, y_pred_clean, average=avg, zero_division=0))

    # Matthews Correlation Coefficient
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true_clean, y_pred_clean))
    except Exception:
        metrics["mcc"] = np.nan

    # Per-class metrics
    if class_names is None:
        class_names = sorted(list(set(y_true_clean) | set(y_pred_clean)))
    else:
        class_names = [str(c) for c in class_names]

    observed_classes = sorted(list(set(y_true_clean) | set(y_pred_clean)))
    missing_observed = sorted(set(observed_classes) - set(class_names))
    if missing_observed:
        warning = (
            "Observed test labels/predictions are missing from the model class list: "
            f"{missing_observed}. Including them for discrete metrics; probability metrics "
            "will only use available probability columns."
        )
        logger.warning(warning)
        _add_warning(warning)

    metric_class_names = _ordered_union(class_names, observed_classes)
    observed_test_classes = set(y_true_clean.tolist())
    classes_absent_from_test = [
        cls for cls in class_names if cls not in observed_test_classes
    ]
    if classes_absent_from_test:
        unique_absent = _ordered_union(classes_absent_from_test)
        absent_preview = unique_absent[:20]
        absent_suffix = ""
        if len(unique_absent) > len(absent_preview):
            absent_suffix = f" (+{len(unique_absent) - len(absent_preview)} more)"
        warning = (
            "Model class list includes classes absent from the test labels: "
            f"{absent_preview}{absent_suffix}. Decision metrics (sensitivity/specificity/PPV/NPV) "
            "are macro-averaged over classes with test support."
        )
        logger.warning(warning)
        _add_warning(warning)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=metric_class_names, average=None, zero_division=0
    )

    per_class = []
    for i, cls in enumerate(metric_class_names):
        per_class.append({
            "class": str(cls),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=metric_class_names)
    metrics["confusion_matrix"] = {
        "labels": [str(c) for c in metric_class_names],
        "matrix": cm.tolist(),
    }

    decision_metrics = compute_decision_metrics(
        y_true_clean,
        y_pred_clean,
        metric_class_names,
        averaging="observed",
    )
    metrics.update(decision_metrics)
    metrics["recall"] = decision_metrics.get("sensitivity")
    metrics["precision"] = decision_metrics.get("ppv")

    # ROC AUC (if probabilities provided)
    if y_proba is not None:
        y_proba_clean = y_proba[valid_mask]
        proba_eval_mask = np.isin(y_true_clean, class_names)
        if not proba_eval_mask.any():
            warning = (
                "Skipping probability metrics because no observed test labels are present "
                f"in the model class list: class_names={class_names}"
            )
            logger.warning(warning)
            _add_warning(warning)
        elif y_proba_clean.shape[1] == len(class_names):
            if not proba_eval_mask.all():
                excluded = int((~proba_eval_mask).sum())
                warning = (
                    "Excluding samples with labels absent from the model class list "
                    f"from probability metrics: n_excluded={excluded}"
                )
                logger.warning(warning)
                _add_warning(warning)
            roc_metrics = compute_roc_auc(
                y_true_clean[proba_eval_mask],
                y_proba_clean[proba_eval_mask],
                class_names,
            )
            metrics["roc_auc"] = roc_metrics
        else:
            warning = (
                "Skipping ROC AUC because probability columns do not match the model "
                f"class list: y_proba_columns={y_proba_clean.shape[1]} "
                f"class_names={len(class_names)}"
            )
            logger.warning(warning)
            _add_warning(warning)

    # Log loss (if probabilities provided)
    if y_proba is not None:
        try:
            y_proba_clean = y_proba[valid_mask]
            keep_idx = np.flatnonzero(np.isin(y_true_clean, class_names))

            if len(keep_idx) > 0:
                metrics["log_loss"] = float(
                    log_loss(
                        y_true_clean[keep_idx],
                        y_proba_clean[keep_idx],
                        labels=class_names,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to compute log loss: {e}")
            metrics["log_loss"] = np.nan

    if metrics_warnings:
        metrics["warnings"] = metrics_warnings

    return metrics


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute ROC AUC scores (OvR style for multiclass).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    class_names : List[str]
        Class names

    Returns
    -------
    roc_metrics : Dict[str, Any]
        Dictionary with:
        - per_class: List of {class, auc} dicts
        - macro: macro-average AUC
        - micro: micro-average AUC (multiclass only)
    """
    roc_metrics = {}
    n_classes = len(class_names)

    if n_classes < 2:
        return {"error": "Need at least 2 classes for ROC AUC"}

    # Ensure y_proba has correct shape
    if y_proba.shape[1] != n_classes:
        return {"error": f"Probability matrix has {y_proba.shape[1]} columns but {n_classes} classes"}

    # Binarize labels
    y_bin = label_binarize(y_true, classes=class_names)

    # Handle binary case (sklearn returns (n, 1) for binary)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # Per-class AUC
    per_class = []
    valid_aucs = []

    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            # No positive samples for this class
            per_class.append({
                "class": str(cls),
                "auc": np.nan,
                "note": "No positive samples in test set",
            })
        elif np.unique(y_bin[:, i]).size < 2:
            per_class.append({
                "class": str(cls),
                "auc": np.nan,
                "note": "Need both positive and negative samples in test set",
            })
        else:
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                per_class.append({
                    "class": str(cls),
                    "auc": float(auc_score),
                })
                valid_aucs.append(auc_score)
            except Exception as e:
                per_class.append({
                    "class": str(cls),
                    "auc": np.nan,
                    "note": f"Error: {str(e)}",
                })

    roc_metrics["per_class"] = per_class

    # Macro-average (average of per-class AUCs)
    if valid_aucs:
        roc_metrics["macro"] = float(np.mean(valid_aucs))
    else:
        roc_metrics["macro"] = np.nan

    # Micro-average (only for multiclass)
    if n_classes > 2:
        try:
            if np.unique(y_bin.ravel()).size < 2:
                roc_metrics["micro"] = np.nan
                roc_metrics["micro_note"] = (
                    "Need both positive and negative labels for micro-average AUC"
                )
            else:
                fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
                roc_metrics["micro"] = float(auc(fpr_micro, tpr_micro))
        except Exception as e:
            logger.warning(f"Failed to compute micro-average AUC: {e}")
            roc_metrics["micro"] = np.nan

    return roc_metrics


def compute_binary_task_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics for a single binary task.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1)
    y_score : np.ndarray
        Predicted scores
    threshold : float
        Classification threshold

    Returns
    -------
    metrics : Dict[str, float]
        Binary classification metrics
    """
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Precision/recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    # MCC
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["mcc"] = np.nan

    # AUC
    try:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics["auc"] = float(auc(fpr, tpr))
        else:
            metrics["auc"] = np.nan
    except Exception:
        metrics["auc"] = np.nan

    return metrics


def format_metrics_for_display(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Format metrics dictionary as a readable DataFrame.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary from compute_classification_metrics

    Returns
    -------
    df : pd.DataFrame
        Formatted metrics table
    """
    rows = []

    # Overall metrics
    overall_keys = [
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_micro",
        "mcc",
        "log_loss",
    ]

    for key in overall_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
            else:
                value_str = str(value)

            rows.append({
                "Metric": key,
                "Value": value_str,
            })

    # ROC AUC
    if "roc_auc" in metrics:
        roc = metrics["roc_auc"]
        if "macro" in roc:
            rows.append({
                "Metric": "roc_auc_macro",
                "Value": f"{roc['macro']:.4f}" if not np.isnan(roc['macro']) else "N/A",
            })
        if "micro" in roc:
            rows.append({
                "Metric": "roc_auc_micro",
                "Value": f"{roc['micro']:.4f}" if not np.isnan(roc['micro']) else "N/A",
            })

    return pd.DataFrame(rows)
