"""Scorer definitions for GridSearchCV multi-metric scoring."""

from __future__ import annotations

from typing import Dict
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Order for per-split metrics output
SCORER_ORDER = [
    "Accuracy",
    "Precision",
    "F1 Score",
    "MCC",
    "Sensitivity",
    "Specificity",
    "ROC AUC",
    "Balanced Accuracy",
]


def get_scorers() -> Dict[str, any]:
    """
    Get multi-metric scorers for GridSearchCV.

    Returns
    -------
    scorers : Dict[str, Scorer]
        Scorer name -> sklearn scorer
    """
    return {
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score, zero_division=0),
        "F1 Score": make_scorer(f1_score, zero_division=0),
        "MCC": make_scorer(matthews_corrcoef),
        "Sensitivity": make_scorer(recall_score, zero_division=0),  # recall of positive
        "Specificity": make_scorer(recall_score, pos_label=0, zero_division=0),  # recall of negative
        "ROC AUC": make_scorer(roc_auc_score, response_method="predict_proba"),
        "Balanced Accuracy": make_scorer(balanced_accuracy_score),
    }
