"""Plotting utilities for inference results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_roc_curves_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "ROC Curves (One-vs-Rest)",
    max_classes: int = 10,
    figsize: tuple = (10, 8),
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Plot ROC curves for multiclass classification (OvR style).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    class_names : List[str]
        Class names
    output_path : Path
        Output file path
    title : str
        Plot title
    max_classes : int
        Maximum number of per-class curves to show
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved image

    Returns
    -------
    roc_data : Dict[str, Any]
        Dictionary with FPR, TPR, and AUC values
    """
    n_classes = len(class_names)

    # Binarize labels
    y_bin = label_binarize(y_true, classes=class_names)

    # Handle binary case
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # Compute ROC curve for each class
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue  # No positive samples for this class

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=figsize)

    # Plot per-class curves (limit to max_classes)
    classes_to_plot = sorted(fpr_dict.keys())[:max_classes]

    for i in classes_to_plot:
        plt.plot(
            fpr_dict[i],
            tpr_dict[i],
            label=f"{class_names[i]} (AUC={auc_dict[i]:.3f})",
            linewidth=2,
        )

    # Compute and plot micro-average (if multiclass)
    if n_classes > 2:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(
            fpr_micro,
            tpr_micro,
            label=f"Micro-average (AUC={auc_micro:.3f})",
            linestyle="--",
            linewidth=2.5,
            color="navy",
        )

    # Compute and plot macro-average
    if len(fpr_dict) > 0:
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict.keys()]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in fpr_dict.keys():
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

        mean_tpr /= len(fpr_dict)
        auc_macro = auc(all_fpr, mean_tpr)

        plt.plot(
            all_fpr,
            mean_tpr,
            label=f"Macro-average (AUC={auc_macro:.3f})",
            linestyle="--",
            linewidth=2.5,
            color="deeppink",
        )

    # Diagonal line
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curves to {output_path}")

    return {
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "auc": auc_dict,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (10, 8),
    dpi: int = 200,
) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str]
        Class names
    output_path : Path
        Output file path
    title : str
        Plot title
    normalize : bool
        If True, normalize by row (true class)
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved image
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        # Normalize by row (true class)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    else:
        cm_norm = cm

    plt.figure(figsize=figsize)

    # Use seaborn heatmap for better appearance
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        square=True,
    )

    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {output_path}")


def plot_score_distributions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Score Distributions by True Class",
    figsize: tuple = (12, 8),
    dpi: int = 200,
) -> None:
    """
    Plot predicted score distributions by true class.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    class_names : List[str]
        Class names
    output_path : Path
        Output file path
    title : str
        Plot title
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved image
    """
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_classes, 1, figsize=figsize, sharex=True)

    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(class_names):
        ax = axes[i]

        # Get indices for this true class
        mask = (y_true == cls)

        if mask.sum() == 0:
            ax.text(0.5, 0.5, f"No samples for class '{cls}'", ha="center", va="center")
            ax.set_ylabel(cls, fontsize=10, fontweight="bold")
            continue

        # Plot histogram of predicted probabilities for this class
        proba_for_true_class = y_proba[mask, i]

        ax.hist(proba_for_true_class, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(np.mean(proba_for_true_class), color="red", linestyle="--", linewidth=2, label="Mean")
        ax.set_ylabel(cls, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved score distributions to {output_path}")


def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: List[str],
    output_dir: Path,
    prefix: str = "inference",
    max_roc_classes: int = 10,
) -> Dict[str, Path]:
    """
    Generate all standard inference plots.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : Optional[np.ndarray]
        Predicted probabilities
    class_names : List[str]
        Class names
    output_dir : Path
        Output directory
    prefix : str
        Filename prefix
    max_roc_classes : int
        Maximum number of classes to show in ROC plot

    Returns
    -------
    plot_paths : Dict[str, Path]
        Dictionary mapping plot type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # Confusion matrix
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred, class_names, cm_path,
        title="Confusion Matrix (Normalized by True Class)",
        normalize=True,
    )
    plot_paths["confusion_matrix"] = cm_path

    # ROC curves (if probabilities available)
    if y_proba is not None:
        roc_path = output_dir / f"{prefix}_roc_curves.png"
        plot_roc_curves_multiclass(
            y_true, y_proba, class_names, roc_path,
            max_classes=max_roc_classes,
        )
        plot_paths["roc_curves"] = roc_path

        # Score distributions
        dist_path = output_dir / f"{prefix}_score_distributions.png"
        plot_score_distributions(
            y_true, y_proba, class_names, dist_path
        )
        plot_paths["score_distributions"] = dist_path

    return plot_paths
