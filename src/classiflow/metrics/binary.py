"""Binary classification metrics computation."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1)
    scores : np.ndarray
        Predicted scores or probabilities

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of computed metrics
    """
    # Determine threshold
    if scores.min() >= 0 and scores.max() <= 1:
        threshold = 0.5
    else:
        threshold = 0.0

    y_pred = (scores >= threshold).astype(int)

    # ROC AUC
    auc_val = np.nan
    if len(np.unique(y_true)) == 2 and np.std(scores) > 0:
        try:
            auc_val = roc_auc_score(y_true, scores)
        except Exception:
            pass

    # MCC
    mcc_val = np.nan
    if len(np.unique(y_true)) == 2:
        try:
            mcc_val = matthews_corrcoef(y_true, y_pred)
        except Exception:
            pass

    return {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(mcc_val) if mcc_val == mcc_val else np.nan,
        "roc_auc": float(auc_val) if auc_val == auc_val else np.nan,
    }
