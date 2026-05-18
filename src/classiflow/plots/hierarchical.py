"""Plotting utilities for hierarchical classification evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


def _prepare_binary_curve_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize binary curve inputs before sklearn/manual curve computation."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)

    if y_true_arr.size != y_score_arr.size:
        size = min(y_true_arr.size, y_score_arr.size)
        logger.warning(
            "Trimming mismatched curve inputs for %s: y_true=%s y_score=%s -> %s",
            context,
            y_true_arr.size,
            y_score_arr.size,
            size,
        )
        y_true_arr = y_true_arr[:size]
        y_score_arr = y_score_arr[:size]

    finite_mask = np.isfinite(y_score_arr)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logger.warning(
            "Dropping %s non-finite score values for %s before curve computation.",
            dropped,
            context,
        )
        y_true_arr = y_true_arr[finite_mask]
        y_score_arr = y_score_arr[finite_mask]

    if y_true_arr.size == 0:
        raise ValueError(f"No valid samples available for {context}")

    y_true_bin = np.asarray(y_true_arr).astype(np.int8, copy=False)
    unique = np.unique(y_true_bin)
    if unique.size < 2:
        raise ValueError(
            f"{context} requires both positive and negative classes; observed {unique.tolist()}"
        )

    return y_true_bin, y_score_arr


def _manual_binary_clf_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute binary cumulative FP/TP counts without sklearn internals."""
    y_true_bin, y_score_arr = _prepare_binary_curve_inputs(y_true, y_score, context)
    sort_idx = np.argsort(y_score_arr, kind="mergesort")[::-1]
    y_true_sorted = y_true_bin[sort_idx]
    y_score_sorted = y_score_arr[sort_idx]

    tps = np.cumsum(y_true_sorted, dtype=np.float64)
    fps = np.cumsum(1 - y_true_sorted, dtype=np.float64)

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    return fps[threshold_indices], tps[threshold_indices], y_score_sorted[threshold_indices]


def safe_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve with fallback for sklearn edge-case IndexError.

    A small number of sklearn/numpy combinations can raise IndexError inside
    roc_curve for valid 1D arrays; this helper retries via a manual path.
    """
    y_true_bin, y_score_arr = _prepare_binary_curve_inputs(y_true, y_score, context)
    try:
        return roc_curve(y_true_bin, y_score_arr)
    except IndexError as exc:
        logger.warning(
            "Falling back to manual ROC computation for %s due to sklearn IndexError: %s",
            context,
            exc,
        )
        fps, tps, thresholds = _manual_binary_clf_curve(y_true_bin, y_score_arr, context)
        fps = np.r_[0.0, fps]
        tps = np.r_[0.0, tps]
        thresholds = np.r_[np.inf, thresholds]

        negatives = fps[-1]
        positives = tps[-1]
        if negatives <= 0 or positives <= 0:
            raise ValueError(f"Cannot compute ROC for {context}: positives={positives}, negatives={negatives}")
        return fps / negatives, tps / positives, thresholds


def safe_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PR curve with fallback for sklearn edge-case IndexError.
    """
    y_true_bin, y_score_arr = _prepare_binary_curve_inputs(y_true, y_score, context)
    try:
        return precision_recall_curve(y_true_bin, y_score_arr)
    except IndexError as exc:
        logger.warning(
            "Falling back to manual PR computation for %s due to sklearn IndexError: %s",
            context,
            exc,
        )
        fps, tps, thresholds = _manual_binary_clf_curve(y_true_bin, y_score_arr, context)
        positives = tps[-1]
        precision = np.divide(
            tps,
            tps + fps,
            out=np.ones_like(tps, dtype=np.float64),
            where=(tps + fps) != 0,
        )
        recall = tps / positives

        sl = slice(None, None, -1)
        precision = np.r_[precision[sl], 1.0]
        recall = np.r_[recall[sl], 0.0]
        thresholds = thresholds[sl]
        return precision, recall, thresholds


def safe_average_precision_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    context: str,
) -> float:
    """
    Compute AP with fallback when sklearn average_precision_score fails.
    """
    y_true_bin, y_score_arr = _prepare_binary_curve_inputs(y_true, y_score, context)
    try:
        return float(average_precision_score(y_true_bin, y_score_arr))
    except IndexError as exc:
        logger.warning(
            "Falling back to manual AP computation for %s due to sklearn IndexError: %s",
            context,
            exc,
        )
        precision, recall, _ = safe_precision_recall_curve(y_true_bin, y_score_arr, context)
        recall_inc = recall[::-1]
        precision_inc = precision[::-1]
        deltas = np.clip(np.diff(recall_inc), 0.0, None)
        return float(np.sum(deltas * precision_inc[1:]))


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    title: str,
    save_path: Path,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Plot ROC curve(s) for binary or multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed integers)
    y_proba : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    classes : List[str]
        Class names
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : Tuple[int, int]
        Figure size (width, height)

    Examples
    --------
    >>> plot_roc_curve(y_val, y_proba, ["Class0", "Class1"], "ROC Curve", Path("roc.png"))
    """
    n_classes = len(classes)

    plt.figure(figsize=figsize)

    if n_classes == 2:
        # Binary classification
        pos_idx = 1
        scores = y_proba[:, pos_idx]
        y_bin = (y_true == pos_idx).astype(int)

        fpr, tpr, _ = safe_roc_curve(y_bin, scores, context=f"{title} binary ROC")
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"{classes[pos_idx]} (AUC={roc_auc:.3f})")

    else:
        # Multiclass - plot per-class and micro-average
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        # Per-class curves
        for i, cls in enumerate(classes):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            fpr_i, tpr_i, _ = safe_roc_curve(
                y_bin[:, i], y_proba[:, i], context=f"{title} ROC class={cls}"
            )
            roc_auc_i = auc(fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, lw=1.5, label=f"{cls} (AUC={roc_auc_i:.3f})")

        # Micro-average
        fpr_micro, tpr_micro, _ = safe_roc_curve(
            y_bin.ravel(), y_proba.ravel(), context=f"{title} ROC micro-average"
        )
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(
            fpr_micro,
            tpr_micro,
            lw=2,
            linestyle="--",
            label=f"micro-avg (AUC={roc_auc_micro:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved ROC curve to {save_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    title: str,
    save_path: Path,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Plot Precision-Recall curve(s) for binary or multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed integers)
    y_proba : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    classes : List[str]
        Class names
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : Tuple[int, int]
        Figure size (width, height)

    Examples
    --------
    >>> plot_pr_curve(y_val, y_proba, ["Class0", "Class1"], "PR Curve", Path("pr.png"))
    """
    n_classes = len(classes)

    plt.figure(figsize=figsize)

    if n_classes == 2:
        # Binary classification
        pos_idx = 1
        scores = y_proba[:, pos_idx]
        y_bin = (y_true == pos_idx).astype(int)

        prec, rec, _ = safe_precision_recall_curve(y_bin, scores, context=f"{title} binary PR")
        ap = safe_average_precision_score(y_bin, scores, context=f"{title} binary PR")

        plt.plot(rec, prec, lw=2, label=f"{classes[pos_idx]} (AP={ap:.3f})")

    else:
        # Multiclass - plot per-class and micro-average
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        # Per-class curves
        for i, cls in enumerate(classes):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            prec_i, rec_i, _ = safe_precision_recall_curve(
                y_bin[:, i], y_proba[:, i], context=f"{title} PR class={cls}"
            )
            ap_i = safe_average_precision_score(
                y_bin[:, i], y_proba[:, i], context=f"{title} PR class={cls}"
            )
            plt.plot(rec_i, prec_i, lw=1.5, label=f"{cls} (AP={ap_i:.3f})")

        # Micro-average
        prec_micro, rec_micro, _ = safe_precision_recall_curve(
            y_bin.ravel(), y_proba.ravel(), context=f"{title} PR micro-average"
        )
        ap_micro = safe_average_precision_score(
            y_bin.ravel(), y_proba.ravel(), context=f"{title} PR micro-average"
        )
        plt.plot(
            rec_micro,
            prec_micro,
            lw=2,
            linestyle="--",
            label=f"micro-avg (AP={ap_micro:.3f})",
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved PR curve to {save_path}")


def plot_averaged_roc_curves(
    all_fpr: List[np.ndarray],
    all_tpr: List[np.ndarray],
    all_aucs: List[float],
    title: str,
    save_path: Path,
    figsize: Tuple[int, int] = (8, 8),
    show_individual: bool = True,
) -> None:
    """
    Plot averaged ROC curves across folds with confidence bands.

    Parameters
    ----------
    all_fpr : List[np.ndarray]
        FPR values for each fold
    all_tpr : List[np.ndarray]
        TPR values for each fold
    all_aucs : List[float]
        AUC values for each fold
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : Tuple[int, int]
        Figure size (width, height)
    show_individual : bool
        Whether to show individual fold curves

    Examples
    --------
    >>> plot_averaged_roc_curves(fprs, tprs, aucs, "Averaged ROC", Path("avg_roc.png"))
    """
    plt.figure(figsize=figsize)

    # Interpolate to common x-axis
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr_interp = []

    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure starts at 0
        all_tpr_interp.append(interp_tpr)

    mean_tpr = np.mean(all_tpr_interp, axis=0)
    std_tpr = np.std(all_tpr_interp, axis=0)
    mean_tpr[-1] = 1.0  # Ensure ends at 1

    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)

    # Plot mean curve
    plt.plot(
        mean_fpr,
        mean_tpr,
        lw=2.5,
        color="b",
        label=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})",
    )

    # Plot std band
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color="b",
        alpha=0.2,
        label="± 1 std. dev.",
    )

    # Plot individual fold curves
    if show_individual:
        for i, (fpr, tpr, auc_val) in enumerate(zip(all_fpr, all_tpr, all_aucs)):
            plt.plot(
                fpr,
                tpr,
                lw=0.8,
                alpha=0.3,
                label=f"Fold {i+1} (AUC={auc_val:.3f})",
            )

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved averaged ROC curve to {save_path}")


def plot_averaged_pr_curves(
    all_rec: List[np.ndarray],
    all_prec: List[np.ndarray],
    all_aps: List[float],
    title: str,
    save_path: Path,
    figsize: Tuple[int, int] = (8, 8),
    show_individual: bool = True,
) -> None:
    """
    Plot averaged PR curves across folds with confidence bands.

    Parameters
    ----------
    all_rec : List[np.ndarray]
        Recall values for each fold
    all_prec : List[np.ndarray]
        Precision values for each fold
    all_aps : List[float]
        Average precision values for each fold
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : Tuple[int, int]
        Figure size (width, height)
    show_individual : bool
        Whether to show individual fold curves

    Examples
    --------
    >>> plot_averaged_pr_curves(recs, precs, aps, "Averaged PR", Path("avg_pr.png"))
    """
    plt.figure(figsize=figsize)

    # Interpolate to common x-axis
    mean_rec = np.linspace(0, 1, 100)
    all_prec_interp = []

    for rec, prec in zip(all_rec, all_prec):
        # Reverse for interpolation (recall is decreasing in sklearn output)
        interp_prec = np.interp(mean_rec, rec[::-1], prec[::-1])
        all_prec_interp.append(interp_prec)

    mean_prec = np.mean(all_prec_interp, axis=0)
    std_prec = np.std(all_prec_interp, axis=0)

    mean_ap = np.mean(all_aps)
    std_ap = np.std(all_aps)

    # Plot mean curve
    plt.plot(
        mean_rec,
        mean_prec,
        lw=2.5,
        color="b",
        label=f"Mean PR (AP={mean_ap:.3f} ± {std_ap:.3f})",
    )

    # Plot std band
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    plt.fill_between(
        mean_rec,
        prec_lower,
        prec_upper,
        color="b",
        alpha=0.2,
        label="± 1 std. dev.",
    )

    # Plot individual fold curves
    if show_individual:
        for i, (rec, prec, ap_val) in enumerate(zip(all_rec, all_prec, all_aps)):
            plt.plot(
                rec,
                prec,
                lw=0.8,
                alpha=0.3,
                label=f"Fold {i+1} (AP={ap_val:.3f})",
            )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved averaged PR curve to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str,
    save_path: Path,
    figsize: Optional[Tuple[int, int]] = None,
    normalize: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix with optional normalization.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed integers)
    y_pred : np.ndarray
        Predicted labels (0-indexed integers)
    classes : List[str]
        Class names
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : Optional[Tuple[int, int]]
        Figure size (auto-computed if None)
    normalize : Optional[str]
        Normalization mode: None, "true", "pred", or "all"

    Examples
    --------
    >>> plot_confusion_matrix(y_true, y_pred, ["A", "B", "C"], "CM", Path("cm.png"))
    """
    # Auto-compute figure size based on number of classes
    if figsize is None:
        n = len(classes)
        size = max(6, min(12, n * 0.8))
        figsize = (size, size)

    # Ensure confusion matrix includes all classes, even if not present in y_true/y_pred
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # Create display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_norm = cm.astype("float")
        if normalize == "true":
            cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm_norm = cm_norm / cm_norm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm_norm = cm_norm / cm_norm.sum()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=classes,
        )

    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved confusion matrix to {save_path}")


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    title: str,
    save_path: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot feature importance bar chart.

    Parameters
    ----------
    importance : np.ndarray
        Feature importance scores
    feature_names : List[str]
        Feature names
    title : str
        Plot title
    save_path : Path
        Path to save figure
    top_n : int
        Number of top features to show
    figsize : Tuple[int, int]
        Figure size (width, height)

    Examples
    --------
    >>> plot_feature_importance(importance, features, "Top Features", Path("fi.png"))
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[indices]
    top_names = [feature_names[i] for i in indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_names)), top_importance, color="steelblue", alpha=0.8)
    plt.yticks(range(len(top_names)), top_names, fontsize=10)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved feature importance plot to {save_path}")


def extract_feature_importance_mlp(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 10,
) -> np.ndarray:
    """
    Extract feature importance using permutation importance for MLP.

    Parameters
    ----------
    model : TorchMLPWrapper
        Trained MLP model
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    feature_names : List[str]
        Feature names
    n_permutations : int
        Number of permutations per feature

    Returns
    -------
    np.ndarray
        Feature importance scores (higher = more important)

    Examples
    --------
    >>> importance = extract_feature_importance_mlp(model, X_val, y_val, features)
    """
    from sklearn.metrics import accuracy_score

    # Baseline performance
    y_pred = model.predict(X_val)
    baseline_acc = accuracy_score(y_val, y_pred)

    importance = np.zeros(X_val.shape[1])

    for i in range(X_val.shape[1]):
        scores = []
        for _ in range(n_permutations):
            X_permuted = X_val.copy()
            # Permute feature i
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            y_pred_perm = model.predict(X_permuted)
            perm_acc = accuracy_score(y_val, y_pred_perm)
            scores.append(baseline_acc - perm_acc)

        importance[i] = np.mean(scores)

    logger.debug(
        f"Computed feature importance for {len(feature_names)} features "
        f"with {n_permutations} permutations"
    )

    return importance
