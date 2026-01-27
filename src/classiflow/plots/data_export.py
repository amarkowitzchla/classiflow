"""Plot data export for interactive UI visualization.

This module provides functions to generate JSON plot data files alongside
PNG images, enabling interactive ROC/PR curve visualization in the UI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

import classiflow
from classiflow.plots.schemas import (
    CurveData,
    PlotCurve,
    PlotKey,
    PlotManifest,
    PlotMetadata,
    PlotScope,
    PlotSummary,
    PlotType,
    TaskType,
)

logger = logging.getLogger(__name__)


def _numpy_to_list(arr: np.ndarray) -> List[float]:
    """Convert numpy array to list of floats, handling NaN/Inf."""
    result = arr.tolist()
    return [float(x) if np.isfinite(x) else 0.0 for x in result]


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    run_id: str,
    scope: PlotScope = PlotScope.INFERENCE,
    fold: Optional[int] = None,
    include_thresholds: bool = True,
) -> PlotCurve:
    """
    Compute ROC curve data for binary or multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed integers)
    y_proba : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    classes : List[str]
        Class names
    run_id : str
        Run identifier
    scope : PlotScope
        Scope of the plot data
    fold : Optional[int]
        Fold number if scope is 'fold'
    include_thresholds : bool
        Whether to include threshold values

    Returns
    -------
    PlotCurve
        ROC curve data ready for JSON serialization
    """
    n_classes = len(classes)
    is_binary = n_classes == 2
    curves: List[CurveData] = []
    auc_values: Dict[str, float] = {}

    if is_binary:
        # Binary classification
        pos_idx = 1
        scores = y_proba[:, pos_idx] if y_proba.ndim > 1 else y_proba
        y_bin = (y_true == pos_idx).astype(int)

        fpr, tpr, thresholds = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)

        curves.append(CurveData(
            label=classes[pos_idx],
            x=_numpy_to_list(fpr),
            y=_numpy_to_list(tpr),
            thresholds=_numpy_to_list(thresholds) if include_thresholds else None,
        ))
        auc_values[classes[pos_idx]] = float(roc_auc)

    else:
        # Multiclass - compute per-class and micro-average
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        # Per-class curves
        for i, cls in enumerate(classes):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            fpr_i, tpr_i, thresholds_i = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_i = auc(fpr_i, tpr_i)

            curves.append(CurveData(
                label=cls,
                x=_numpy_to_list(fpr_i),
                y=_numpy_to_list(tpr_i),
                thresholds=_numpy_to_list(thresholds_i) if include_thresholds else None,
            ))
            auc_values[cls] = float(roc_auc_i)

        # Micro-average
        fpr_micro, tpr_micro, thresholds_micro = roc_curve(y_bin.ravel(), y_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        curves.append(CurveData(
            label="micro",
            x=_numpy_to_list(fpr_micro),
            y=_numpy_to_list(tpr_micro),
            thresholds=_numpy_to_list(thresholds_micro) if include_thresholds else None,
        ))
        auc_values["micro"] = float(roc_auc_micro)

        # Macro-average (computed by interpolation)
        all_fpr = np.unique(np.concatenate([
            curves[i].x for i in range(len(classes))
            if i < len(curves) and curves[i].label in classes
        ]))
        if len(all_fpr) > 0:
            mean_tpr = np.zeros_like(all_fpr)
            count = 0
            for curve in curves:
                if curve.label in classes:
                    mean_tpr += np.interp(all_fpr, curve.x, curve.y)
                    count += 1
            if count > 0:
                mean_tpr /= count
                roc_auc_macro = auc(all_fpr, mean_tpr)
                curves.append(CurveData(
                    label="macro",
                    x=_numpy_to_list(all_fpr),
                    y=_numpy_to_list(mean_tpr),
                ))
                auc_values["macro"] = float(roc_auc_macro)

    return PlotCurve(
        plot_type=PlotType.ROC,
        scope=scope,
        task=TaskType.BINARY if is_binary else TaskType.MULTICLASS,
        labels=classes,
        curves=curves,
        summary=PlotSummary(auc=auc_values),
        metadata=PlotMetadata(
            generated_at=datetime.now(),
            source="internal",
            classiflow_version=classiflow.__version__,
            run_id=run_id,
            fold=fold,
        ),
    )


def compute_pr_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    run_id: str,
    scope: PlotScope = PlotScope.INFERENCE,
    fold: Optional[int] = None,
    include_thresholds: bool = True,
) -> PlotCurve:
    """
    Compute PR curve data for binary or multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0-indexed integers)
    y_proba : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    classes : List[str]
        Class names
    run_id : str
        Run identifier
    scope : PlotScope
        Scope of the plot data
    fold : Optional[int]
        Fold number if scope is 'fold'
    include_thresholds : bool
        Whether to include threshold values

    Returns
    -------
    PlotCurve
        PR curve data ready for JSON serialization
    """
    n_classes = len(classes)
    is_binary = n_classes == 2
    curves: List[CurveData] = []
    ap_values: Dict[str, float] = {}

    if is_binary:
        # Binary classification
        pos_idx = 1
        scores = y_proba[:, pos_idx] if y_proba.ndim > 1 else y_proba
        y_bin = (y_true == pos_idx).astype(int)

        prec, rec, thresholds = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)

        curves.append(CurveData(
            label=classes[pos_idx],
            x=_numpy_to_list(rec),
            y=_numpy_to_list(prec),
            thresholds=_numpy_to_list(np.append(thresholds, [0.0])) if include_thresholds else None,
        ))
        ap_values[classes[pos_idx]] = float(ap)

    else:
        # Multiclass - compute per-class and micro-average
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        # Per-class curves
        for i, cls in enumerate(classes):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            prec_i, rec_i, thresholds_i = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap_i = average_precision_score(y_bin[:, i], y_proba[:, i])

            curves.append(CurveData(
                label=cls,
                x=_numpy_to_list(rec_i),
                y=_numpy_to_list(prec_i),
                thresholds=_numpy_to_list(np.append(thresholds_i, [0.0])) if include_thresholds else None,
            ))
            ap_values[cls] = float(ap_i)

        # Micro-average
        prec_micro, rec_micro, thresholds_micro = precision_recall_curve(
            y_bin.ravel(), y_proba.ravel()
        )
        ap_micro = average_precision_score(y_bin, y_proba, average="micro")
        curves.append(CurveData(
            label="micro",
            x=_numpy_to_list(rec_micro),
            y=_numpy_to_list(prec_micro),
            thresholds=_numpy_to_list(np.append(thresholds_micro, [0.0])) if include_thresholds else None,
        ))
        ap_values["micro"] = float(ap_micro)

    return PlotCurve(
        plot_type=PlotType.PR,
        scope=scope,
        task=TaskType.BINARY if is_binary else TaskType.MULTICLASS,
        labels=classes,
        curves=curves,
        summary=PlotSummary(ap=ap_values),
        metadata=PlotMetadata(
            generated_at=datetime.now(),
            source="internal",
            classiflow_version=classiflow.__version__,
            run_id=run_id,
            fold=fold,
        ),
    )


def compute_averaged_roc_data(
    all_fpr: List[np.ndarray],
    all_tpr: List[np.ndarray],
    all_aucs: List[float],
    classes: List[str],
    run_id: str,
) -> PlotCurve:
    """
    Compute averaged ROC curve data across folds with confidence bands.

    Parameters
    ----------
    all_fpr : List[np.ndarray]
        FPR values for each fold
    all_tpr : List[np.ndarray]
        TPR values for each fold
    all_aucs : List[float]
        AUC values for each fold
    classes : List[str]
        Class names
    run_id : str
        Run identifier

    Returns
    -------
    PlotCurve
        Averaged ROC curve data with fold curves and std bands
    """
    # Interpolate to common x-axis
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr_interp = []

    fold_curves: List[CurveData] = []
    for i, (fpr, tpr, auc_val) in enumerate(zip(all_fpr, all_tpr, all_aucs)):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        all_tpr_interp.append(interp_tpr)

        fold_curves.append(CurveData(
            label=f"Fold {i + 1}",
            x=_numpy_to_list(fpr),
            y=_numpy_to_list(tpr),
        ))

    mean_tpr = np.mean(all_tpr_interp, axis=0)
    std_tpr = np.std(all_tpr_interp, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = float(np.mean(all_aucs))
    std_auc = float(np.std(all_aucs))

    # Mean curve
    curves = [CurveData(
        label="mean",
        x=_numpy_to_list(mean_fpr),
        y=_numpy_to_list(mean_tpr),
    )]

    # Std band
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    return PlotCurve(
        plot_type=PlotType.ROC,
        scope=PlotScope.AVERAGED,
        task=TaskType.BINARY if len(classes) == 2 else TaskType.MULTICLASS,
        labels=classes,
        curves=curves,
        summary=PlotSummary(auc={"mean": mean_auc, "std": std_auc}),
        metadata=PlotMetadata(
            generated_at=datetime.now(),
            source="internal",
            classiflow_version=classiflow.__version__,
            run_id=run_id,
        ),
        std_band={
            "x": _numpy_to_list(mean_fpr),
            "y_upper": _numpy_to_list(tpr_upper),
            "y_lower": _numpy_to_list(tpr_lower),
        },
        fold_curves=fold_curves,
        fold_metrics={"auc": [float(a) for a in all_aucs]},
    )


def compute_averaged_pr_data(
    all_rec: List[np.ndarray],
    all_prec: List[np.ndarray],
    all_aps: List[float],
    classes: List[str],
    run_id: str,
) -> PlotCurve:
    """
    Compute averaged PR curve data across folds with confidence bands.

    Parameters
    ----------
    all_rec : List[np.ndarray]
        Recall values for each fold
    all_prec : List[np.ndarray]
        Precision values for each fold
    all_aps : List[float]
        Average precision values for each fold
    classes : List[str]
        Class names
    run_id : str
        Run identifier

    Returns
    -------
    PlotCurve
        Averaged PR curve data with fold curves and std bands
    """
    # Interpolate to common x-axis
    mean_rec = np.linspace(0, 1, 100)
    all_prec_interp = []

    fold_curves: List[CurveData] = []
    for i, (rec, prec, ap_val) in enumerate(zip(all_rec, all_prec, all_aps)):
        # Reverse for interpolation (recall decreases in sklearn output)
        interp_prec = np.interp(mean_rec, rec[::-1], prec[::-1])
        all_prec_interp.append(interp_prec)

        fold_curves.append(CurveData(
            label=f"Fold {i + 1}",
            x=_numpy_to_list(rec),
            y=_numpy_to_list(prec),
        ))

    mean_prec = np.mean(all_prec_interp, axis=0)
    std_prec = np.std(all_prec_interp, axis=0)

    mean_ap = float(np.mean(all_aps))
    std_ap = float(np.std(all_aps))

    # Mean curve
    curves = [CurveData(
        label="mean",
        x=_numpy_to_list(mean_rec),
        y=_numpy_to_list(mean_prec),
    )]

    # Std band
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)

    return PlotCurve(
        plot_type=PlotType.PR,
        scope=PlotScope.AVERAGED,
        task=TaskType.BINARY if len(classes) == 2 else TaskType.MULTICLASS,
        labels=classes,
        curves=curves,
        summary=PlotSummary(ap={"mean": mean_ap, "std": std_ap}),
        metadata=PlotMetadata(
            generated_at=datetime.now(),
            source="internal",
            classiflow_version=classiflow.__version__,
            run_id=run_id,
        ),
        std_band={
            "x": _numpy_to_list(mean_rec),
            "y_upper": _numpy_to_list(prec_upper),
            "y_lower": _numpy_to_list(prec_lower),
        },
        fold_curves=fold_curves,
        fold_metrics={"ap": [float(a) for a in all_aps]},
    )


def save_plot_data(
    plot_data: PlotCurve,
    output_path: Path,
) -> None:
    """
    Save plot data to JSON file.

    Parameters
    ----------
    plot_data : PlotCurve
        Plot data to save
    output_path : Path
        Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            plot_data.model_dump(mode="json"),
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Saved plot data to {output_path}")


def create_plot_manifest(
    run_dir: Path,
    run_id: str,
    available_plots: Dict[str, str],
    fallback_pngs: Optional[Dict[str, str]] = None,
) -> PlotManifest:
    """
    Create and save a plot manifest file.

    Parameters
    ----------
    run_dir : Path
        Run directory
    run_id : str
        Run identifier
    available_plots : Dict[str, str]
        Mapping of plot key to relative JSON file path
    fallback_pngs : Dict[str, str], optional
        Mapping of plot key to fallback PNG file path

    Returns
    -------
    PlotManifest
        The created manifest
    """
    manifest = PlotManifest(
        available=available_plots,
        fallback_pngs=fallback_pngs or {},
        generated_at=datetime.now(),
        classiflow_version=classiflow.__version__,
    )

    manifest_path = run_dir / "plots" / "plot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(
            manifest.model_dump(mode="json"),
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Saved plot manifest to {manifest_path}")
    return manifest


def generate_technical_validation_plots(
    run_dir: Path,
    run_id: str,
    fold_data: List[Dict[str, Any]],
    classes: List[str],
) -> PlotManifest:
    """
    Generate all plot JSON files for a technical validation run.

    Parameters
    ----------
    run_dir : Path
        Run directory
    run_id : str
        Run identifier
    fold_data : List[Dict[str, Any]]
        List of fold data dictionaries, each containing:
        - y_true: np.ndarray
        - y_proba: np.ndarray
        - fold_num: int
    classes : List[str]
        Class names

    Returns
    -------
    PlotManifest
        Manifest of generated plots
    """
    plots_dir = run_dir / "plots"
    available: Dict[str, str] = {}
    fallback_pngs: Dict[str, str] = {}

    # Collect fold-level ROC/PR data
    all_fpr, all_tpr, all_aucs = [], [], []
    all_rec, all_prec, all_aps = [], [], []
    fold_roc_curves = []
    fold_pr_curves = []

    for fold_info in fold_data:
        y_true = fold_info["y_true"]
        y_proba = fold_info["y_proba"]
        fold_num = fold_info["fold_num"]

        # Compute ROC for this fold
        roc_data = compute_roc_curve_data(
            y_true, y_proba, classes, run_id,
            scope=PlotScope.FOLD, fold=fold_num,
        )
        fold_roc_curves.append(roc_data)

        # Compute PR for this fold
        pr_data = compute_pr_curve_data(
            y_true, y_proba, classes, run_id,
            scope=PlotScope.FOLD, fold=fold_num,
        )
        fold_pr_curves.append(pr_data)

        # Extract data for averaging (using binary or micro-average curve)
        if len(classes) == 2:
            # Binary: use the positive class curve
            roc_curve_data = roc_data.curves[0]
            pr_curve_data = pr_data.curves[0]
            auc_key = classes[1]
            ap_key = classes[1]
        else:
            # Multiclass: use micro-average
            roc_curve_data = next((c for c in roc_data.curves if c.label == "micro"), roc_data.curves[0])
            pr_curve_data = next((c for c in pr_data.curves if c.label == "micro"), pr_data.curves[0])
            auc_key = "micro"
            ap_key = "micro"

        all_fpr.append(np.array(roc_curve_data.x))
        all_tpr.append(np.array(roc_curve_data.y))
        all_aucs.append(roc_data.summary.auc.get(auc_key, 0.0))

        all_rec.append(np.array(pr_curve_data.x))
        all_prec.append(np.array(pr_curve_data.y))
        all_aps.append(pr_data.summary.ap.get(ap_key, 0.0))

    # Generate averaged plots
    if all_fpr:
        # Averaged ROC
        averaged_roc = compute_averaged_roc_data(all_fpr, all_tpr, all_aucs, classes, run_id)
        save_plot_data(averaged_roc, plots_dir / "roc_averaged.json")
        available[PlotKey.ROC_AVERAGED] = "plots/roc_averaged.json"

        # Averaged PR
        averaged_pr = compute_averaged_pr_data(all_rec, all_prec, all_aps, classes, run_id)
        save_plot_data(averaged_pr, plots_dir / "pr_averaged.json")
        available[PlotKey.PR_AVERAGED] = "plots/pr_averaged.json"

        # Save by-fold data as a combined file
        by_fold_roc = PlotCurve(
            plot_type=PlotType.ROC,
            scope=PlotScope.FOLD,
            task=TaskType.BINARY if len(classes) == 2 else TaskType.MULTICLASS,
            labels=classes,
            curves=[c for fold_data in fold_roc_curves for c in fold_data.curves],
            summary=PlotSummary(auc={"folds": all_aucs}),
            metadata=PlotMetadata(
                generated_at=datetime.now(),
                source="internal",
                classiflow_version=classiflow.__version__,
                run_id=run_id,
            ),
            fold_metrics={"auc": all_aucs},
        )
        save_plot_data(by_fold_roc, plots_dir / "roc_by_fold.json")
        available[PlotKey.ROC_BY_FOLD] = "plots/roc_by_fold.json"

        by_fold_pr = PlotCurve(
            plot_type=PlotType.PR,
            scope=PlotScope.FOLD,
            task=TaskType.BINARY if len(classes) == 2 else TaskType.MULTICLASS,
            labels=classes,
            curves=[c for fold_data in fold_pr_curves for c in fold_data.curves],
            summary=PlotSummary(ap={"folds": all_aps}),
            metadata=PlotMetadata(
                generated_at=datetime.now(),
                source="internal",
                classiflow_version=classiflow.__version__,
                run_id=run_id,
            ),
            fold_metrics={"ap": all_aps},
        )
        save_plot_data(by_fold_pr, plots_dir / "pr_by_fold.json")
        available[PlotKey.PR_BY_FOLD] = "plots/pr_by_fold.json"

    # Check for fallback PNGs
    if (run_dir / "averaged_roc.png").exists():
        fallback_pngs[PlotKey.ROC_AVERAGED] = "averaged_roc.png"
    if (run_dir / "averaged_pr.png").exists():
        fallback_pngs[PlotKey.PR_AVERAGED] = "averaged_pr.png"

    return create_plot_manifest(run_dir, run_id, available, fallback_pngs)


def generate_inference_plots(
    run_dir: Path,
    run_id: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
) -> PlotManifest:
    """
    Generate all plot JSON files for an independent test (inference) run.

    Parameters
    ----------
    run_dir : Path
        Run directory
    run_id : str
        Run identifier
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    classes : List[str]
        Class names

    Returns
    -------
    PlotManifest
        Manifest of generated plots
    """
    plots_dir = run_dir / "plots"
    available: Dict[str, str] = {}
    fallback_pngs: Dict[str, str] = {}

    # ROC curves
    roc_data = compute_roc_curve_data(
        y_true, y_proba, classes, run_id,
        scope=PlotScope.INFERENCE,
    )
    save_plot_data(roc_data, plots_dir / "roc_inference.json")
    available[PlotKey.ROC_INFERENCE] = "plots/roc_inference.json"

    # PR curves
    pr_data = compute_pr_curve_data(
        y_true, y_proba, classes, run_id,
        scope=PlotScope.INFERENCE,
    )
    save_plot_data(pr_data, plots_dir / "pr_inference.json")
    available[PlotKey.PR_INFERENCE] = "plots/pr_inference.json"

    # Check for fallback PNGs
    if (run_dir / "inference_roc_curves.png").exists():
        fallback_pngs[PlotKey.ROC_INFERENCE] = "inference_roc_curves.png"

    return create_plot_manifest(run_dir, run_id, available, fallback_pngs)
