"""Hierarchical-aware metrics computation for multi-level classification."""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_hierarchical_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[np.ndarray] = None,
    hierarchy_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics at each level of hierarchical classification.

    For hierarchical predictions like "TumorTypeA::SubtypeX", computes:
    - L1 metrics (tumor type level)
    - L2 metrics (subtype level, per branch)
    - L3 metrics (if applicable)
    - Leaf metrics (finest level)

    Parameters
    ----------
    y_true : pd.Series
        True labels (can be hierarchical like "L1::L2::L3")
    y_pred : pd.Series
        Predicted labels (hierarchical format)
    y_proba : Optional[np.ndarray]
        Predicted probabilities (if available)
    hierarchy_spec : Optional[Dict[str, Any]]
        Hierarchy specification with level definitions

    Returns
    -------
    metrics : Dict[str, Any]
        Metrics organized by level:
        - overall: leaf-level metrics
        - l1: L1-level metrics
        - l2: L2-level metrics (per branch or aggregated)
        - l3: L3-level metrics (if applicable)
    """
    # Parse hierarchical labels
    y_true_levels = _parse_hierarchical_labels(y_true)
    y_pred_levels = _parse_hierarchical_labels(y_pred)

    max_levels = max(
        max(len(levels) for levels in y_true_levels),
        max(len(levels) for levels in y_pred_levels),
    )

    metrics = {
        "n_samples": len(y_true),
        "n_levels": max_levels,
    }

    # Compute metrics at each level
    for level_idx in range(max_levels):
        level_name = f"l{level_idx + 1}"

        y_true_level = [levels[level_idx] if level_idx < len(levels) else None
                        for levels in y_true_levels]
        y_pred_level = [levels[level_idx] if level_idx < len(levels) else None
                        for levels in y_pred_levels]

        # Filter out None values
        valid_mask = [(yt is not None and yp is not None)
                      for yt, yp in zip(y_true_level, y_pred_level)]
        y_true_level_clean = [yt for yt, valid in zip(y_true_level, valid_mask) if valid]
        y_pred_level_clean = [yp for yp, valid in zip(y_pred_level, valid_mask) if valid]

        if len(y_true_level_clean) > 0:
            level_metrics = _compute_level_metrics(
                y_true_level_clean,
                y_pred_level_clean,
                level_name=level_name,
            )
            metrics[level_name] = level_metrics
        else:
            logger.warning(f"No valid samples for level {level_name}")
            metrics[level_name] = {}

    # Overall metrics (leaf level = full hierarchical path)
    metrics["overall"] = _compute_level_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        level_name="leaf",
    )

    return metrics


def _parse_hierarchical_labels(labels: pd.Series) -> List[List[str]]:
    """
    Parse hierarchical labels into level components.

    Examples:
        "TypeA::SubtypeX" -> ["TypeA", "SubtypeX"]
        "TypeA" -> ["TypeA"]

    Parameters
    ----------
    labels : pd.Series
        Label series

    Returns
    -------
    parsed : List[List[str]]
        List of level components for each sample
    """
    parsed = []
    for label in labels:
        if pd.isna(label):
            parsed.append([])
        else:
            label_str = str(label)
            if "::" in label_str:
                parsed.append(label_str.split("::"))
            else:
                parsed.append([label_str])
    return parsed


def _compute_level_metrics(
    y_true: List[str],
    y_pred: List[str],
    level_name: str = "level",
) -> Dict[str, Any]:
    """
    Compute metrics for a single hierarchical level.

    Parameters
    ----------
    y_true : List[str]
        True labels at this level
    y_pred : List[str]
        Predicted labels at this level
    level_name : str
        Name of level for logging

    Returns
    -------
    metrics : Dict[str, Any]
        Metrics dictionary with overall + per-class
    """
    if len(y_true) == 0:
        return {"error": "No samples"}

    # Get unique classes
    classes = sorted(list(set(y_true) | set(y_pred)))

    metrics = {
        "n_samples": len(y_true),
        "n_classes": len(classes),
    }

    # Overall metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Macro F1
    metrics["f1_macro"] = float(f1_score(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    ))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )

    per_class = []
    for i, cls in enumerate(classes):
        per_class.append({
            "class": str(cls),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_classes"] = classes

    logger.debug(f"Computed metrics for {level_name}: acc={metrics['accuracy']:.3f}, f1_macro={metrics['f1_macro']:.3f}")

    return metrics


def create_hierarchical_metrics_sheets(
    metrics: Dict[str, Any],
    writer: pd.ExcelWriter,
) -> None:
    """
    Create Excel sheets for hierarchical metrics.

    Creates sheets:
    - Overall_Metrics_L1, Per_Class_L1, Confusion_L1
    - Overall_Metrics_L2, Per_Class_L2, Confusion_L2
    - ... (for each level)
    - Overall_Metrics_Leaf, Per_Class_Leaf, Confusion_Leaf

    Parameters
    ----------
    metrics : Dict[str, Any]
        Hierarchical metrics from compute_hierarchical_metrics
    writer : pd.ExcelWriter
        Excel writer object
    """
    n_levels = metrics.get("n_levels", 0)

    # Process each level
    for level_idx in range(n_levels):
        level_name = f"l{level_idx + 1}"
        if level_name not in metrics:
            continue

        level_metrics = metrics[level_name]
        if not level_metrics or "error" in level_metrics:
            continue

        level_suffix = f"L{level_idx + 1}"

        # Overall metrics sheet
        _write_overall_metrics_sheet(
            level_metrics,
            writer,
            sheet_name=f"Overall_Metrics_{level_suffix}",
        )

        # Per-class metrics sheet
        _write_per_class_sheet(
            level_metrics,
            writer,
            sheet_name=f"Per_Class_{level_suffix}",
        )

        # Confusion matrix sheet
        _write_confusion_matrix_sheet(
            level_metrics,
            writer,
            sheet_name=f"Confusion_{level_suffix}",
        )

    # Leaf (overall) metrics
    if "overall" in metrics and "error" not in metrics["overall"]:
        _write_overall_metrics_sheet(
            metrics["overall"],
            writer,
            sheet_name="Overall_Metrics_Leaf",
        )
        _write_per_class_sheet(
            metrics["overall"],
            writer,
            sheet_name="Per_Class_Leaf",
        )
        _write_confusion_matrix_sheet(
            metrics["overall"],
            writer,
            sheet_name="Confusion_Leaf",
        )

    logger.info(f"Created hierarchical metrics sheets for {n_levels} levels")


def _write_overall_metrics_sheet(
    level_metrics: Dict[str, Any],
    writer: pd.ExcelWriter,
    sheet_name: str,
) -> None:
    """Write overall metrics to Excel sheet."""
    overall_data = {
        "Metric": ["n_samples", "n_classes", "accuracy", "f1_macro"],
        "Value": [
            level_metrics.get("n_samples", 0),
            level_metrics.get("n_classes", 0),
            level_metrics.get("accuracy", 0.0),
            level_metrics.get("f1_macro", 0.0),
        ],
    }

    df = pd.DataFrame(overall_data)
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_per_class_sheet(
    level_metrics: Dict[str, Any],
    writer: pd.ExcelWriter,
    sheet_name: str,
) -> None:
    """Write per-class metrics to Excel sheet."""
    if "per_class" not in level_metrics:
        return

    df = pd.DataFrame(level_metrics["per_class"])
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_confusion_matrix_sheet(
    level_metrics: Dict[str, Any],
    writer: pd.ExcelWriter,
    sheet_name: str,
    max_classes: int = 50,
) -> None:
    """Write confusion matrix to Excel sheet."""
    if "confusion_matrix" not in level_metrics:
        return

    cm = np.array(level_metrics["confusion_matrix"])
    classes = level_metrics.get("confusion_matrix_classes", [])

    # Cap size for readability
    if len(classes) > max_classes:
        logger.warning(f"Confusion matrix too large ({len(classes)} classes), truncating to {max_classes}")
        cm = cm[:max_classes, :max_classes]
        classes = classes[:max_classes]

    # Create DataFrame
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"

    cm_df.to_excel(writer, sheet_name=sheet_name)


def summarize_hierarchical_metrics(
    metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create summary table of hierarchical metrics across levels.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Hierarchical metrics

    Returns
    -------
    summary : pd.DataFrame
        Summary table with rows per level, columns for key metrics
    """
    rows = []

    n_levels = metrics.get("n_levels", 0)
    for level_idx in range(n_levels):
        level_name = f"l{level_idx + 1}"
        if level_name in metrics and "error" not in metrics[level_name]:
            lm = metrics[level_name]
            rows.append({
                "Level": f"L{level_idx + 1}",
                "N_Samples": lm.get("n_samples", 0),
                "N_Classes": lm.get("n_classes", 0),
                "Accuracy": lm.get("accuracy", 0.0),
                "F1_Macro": lm.get("f1_macro", 0.0),
            })

    # Add leaf
    if "overall" in metrics and "error" not in metrics["overall"]:
        om = metrics["overall"]
        rows.append({
            "Level": "Leaf",
            "N_Samples": om.get("n_samples", 0),
            "N_Classes": om.get("n_classes", 0),
            "Accuracy": om.get("accuracy", 0.0),
            "F1_Macro": om.get("f1_macro", 0.0),
        })

    return pd.DataFrame(rows)
