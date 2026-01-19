"""Metrics computation for inference evaluation."""

from __future__ import annotations

import logging
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
    y_true_clean = np.array(y_true)[valid_mask]
    y_pred_clean = np.array(y_pred)[valid_mask]

    if len(y_true_clean) == 0:
        return {"error": "No valid samples for evaluation"}

    n_samples = len(y_true_clean)
    metrics["n_samples"] = n_samples

    # Overall metrics
    metrics["accuracy"] = float(accuracy_score(y_true_clean, y_pred_clean))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_clean, y_pred_clean))

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

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=class_names, average=None, zero_division=0
    )

    per_class = []
    for i, cls in enumerate(class_names):
        per_class.append({
            "class": str(cls),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=class_names)
    metrics["confusion_matrix"] = {
        "labels": [str(c) for c in class_names],
        "matrix": cm.tolist(),
    }

    decision_metrics = compute_decision_metrics(y_true_clean, y_pred_clean, class_names)
    metrics.update(decision_metrics)
    metrics["recall"] = decision_metrics.get("sensitivity")
    metrics["precision"] = decision_metrics.get("ppv")

    # ROC AUC (if probabilities provided)
    if y_proba is not None:
        y_proba_clean = y_proba[valid_mask]
        roc_metrics = compute_roc_auc(y_true_clean, y_proba_clean, class_names)
        metrics["roc_auc"] = roc_metrics

    # Log loss (if probabilities provided)
    if y_proba is not None:
        try:
            y_proba_clean = y_proba[valid_mask]
            # Filter to classes present in y_true
            class_set = set(class_names)
            keep_idx = [i for i, t in enumerate(y_true_clean) if t in class_set]

            if len(keep_idx) > 0:
                metrics["log_loss"] = float(
                    log_loss(
                        [y_true_clean[i] for i in keep_idx],
                        y_proba_clean[keep_idx],
                        labels=class_names,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to compute log loss: {e}")
            metrics["log_loss"] = np.nan

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
