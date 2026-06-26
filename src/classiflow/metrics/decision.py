"""Decision-focused metrics (sensitivity, specificity, PPV/NPV)."""

from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_decision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    averaging: Literal["all", "observed"] = "all",
) -> Dict[str, float]:
    """
    Compute macro-averaged sensitivity, specificity, PPV, and NPV.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : List[str]
        Label order for confusion matrix construction.
    averaging : {"all", "observed"}
        If "all", average over all provided classes.
        If "observed", average only over classes with non-zero support in y_true
        (while still computing each class against the full confusion matrix).
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    total = cm.sum()
    sensitivities = []
    specificities = []
    ppv = []
    npv = []

    if averaging == "observed":
        class_indices = [idx for idx in range(len(class_names)) if cm[idx, :].sum() > 0]
    else:
        class_indices = list(range(len(class_names)))

    for idx in class_indices:
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = total - (tp + fn + fp)

        sensitivities.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
        ppv.append(tp / (tp + fp) if (tp + fp) else 0.0)
        npv.append(tn / (tn + fn) if (tn + fn) else 0.0)

    return {
        "sensitivity": float(np.mean(sensitivities)) if sensitivities else float("nan"),
        "specificity": float(np.mean(specificities)) if specificities else float("nan"),
        "ppv": float(np.mean(ppv)) if ppv else float("nan"),
        "npv": float(np.mean(npv)) if npv else float("nan"),
    }
