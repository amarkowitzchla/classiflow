"""Direct multiclass training with nested cross-validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize

from classiflow.config import MulticlassConfig
from classiflow.io import load_data, load_data_with_groups, validate_data
from classiflow.lineage.hashing import get_file_metadata
from classiflow.lineage.manifest import create_training_manifest
from classiflow.models import AdaptiveSMOTE, get_estimators, get_param_grids, resolve_device
from classiflow.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
)
from classiflow.splitting import (
    iter_outer_splits,
    iter_inner_splits,
    assert_no_patient_leakage,
    make_group_labels,
)

logger = logging.getLogger(__name__)

REFIT_SCORER = "F1 Macro"
MC_SCORER_ORDER = [
    "Accuracy",
    "Balanced Accuracy",
    "F1 Macro",
    "F1 Weighted",
]


def train_multiclass_classifier(config: MulticlassConfig) -> Dict[str, Any]:
    """Train a direct multiclass classifier using nested CV."""
    logger.info("Starting multiclass training")
    data_path = config.resolved_data_path
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Label: {config.label_col}")
    logger.info(f"  SMOTE: {config.smote_mode}")
    logger.info(f"  Device: {config.device}")

    config.outdir.mkdir(parents=True, exist_ok=True)

    groups = None
    if config.patient_col:
        X_full, y_full, groups = load_data_with_groups(
            data_path,
            config.label_col,
            config.patient_col,
            feature_cols=config.feature_cols,
        )
    else:
        X_full, y_full = load_data(data_path, config.label_col, feature_cols=config.feature_cols)

    if config.classes:
        missing = set(config.classes) - set(y_full.unique().tolist())
        if missing:
            raise ValueError(f"Classes not found in data: {sorted(missing)}")
        mask = y_full.isin(config.classes)
        X_full = X_full[mask]
        y_full = y_full[mask]
        if groups is not None:
            groups = groups[mask]
        classes = list(config.classes)
    else:
        classes = sorted(y_full.unique().tolist())

    validate_data(X_full, y_full)

    if config.patient_col and groups is not None:
        patient_df = pd.DataFrame(
            {config.patient_col: groups, "label": y_full.values},
            index=X_full.index,
        )
        make_group_labels(patient_df, config.patient_col, "label")

    y_cat = pd.Categorical(y_full, categories=classes, ordered=True)
    if (y_cat.codes < 0).any():
        raise ValueError("Label encoding failed: unmatched classes detected.")
    y_enc = pd.Series(y_cat.codes, index=y_full.index, name=y_full.name)

    resolved_device = resolve_device(config.device)
    _log_torch_status(config.device, resolved_device)

    file_metadata = get_file_metadata(data_path)
    config_dict = config.to_dict()
    config_dict["resolved_device"] = resolved_device
    config_dict["stratification_level"] = "patient" if config.patient_col else "sample"

    manifest = create_training_manifest(
        data_path=data_path,
        data_hash=file_metadata["sha256_hash"],
        data_size_bytes=file_metadata["size_bytes"],
        data_row_count=file_metadata.get("row_count"),
        config=config_dict,
        task_type="multiclass",
        feature_list=X_full.columns.tolist(),
        task_definitions={"multiclass": f"classes={classes}"},
        hierarchical=False,
    )
    manifest.save(config.outdir / "run.json")
    logger.info(f"Saved training manifest: run_id={manifest.run_id}")

    results = _run_multiclass_nested_cv(
        X_full=X_full,
        y_full=y_enc,
        classes=classes,
        config=config,
        resolved_device=resolved_device,
        groups=groups,
    )

    logger.info("Multiclass training complete")
    return results


def _run_multiclass_nested_cv(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    classes: List[str],
    config: MulticlassConfig,
    resolved_device: str,
    groups: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Run nested CV for direct multiclass training."""
    label_ids = list(range(len(classes)))
    logreg_params = {
        "solver": config.logreg_solver,
        "penalty": config.logreg_penalty,
        "max_iter": config.logreg_max_iter,
        "tol": config.logreg_tol,
        "C": config.logreg_C,
        "class_weight": config.logreg_class_weight,
        "n_jobs": config.logreg_n_jobs,
    }
    if config.logreg_multi_class and config.logreg_multi_class != "auto":
        logreg_params["multi_class"] = config.logreg_multi_class
    estimators = _apply_device_to_estimators(
        get_estimators(
            config.random_state,
            config.max_iter,
            logreg_params=logreg_params,
            resolved_device=resolved_device,
        ),
        resolved_device,
    )
    param_grids = get_param_grids(resolved_device=resolved_device)
    estimators, param_grids = _filter_estimators(
        estimators,
        param_grids,
        config.estimator_mode,
    )
    scorers = _get_multiclass_scorers(label_ids)
    logger.info("Enabled estimators: %s", ", ".join(estimators.keys()))

    variants = ["smote", "none"] if config.smote_mode in ("on", "both") else ["none"]

    df_groups = None
    groups_series = None
    if config.patient_col and groups is not None:
        groups_series = groups if isinstance(groups, pd.Series) else pd.Series(groups, index=X_full.index)
        df_groups = pd.DataFrame({config.patient_col: groups_series}, index=X_full.index)
        outer_splits = iter_outer_splits(
            df=df_groups,
            y=y_full,
            patient_col=config.patient_col,
            n_splits=config.outer_folds,
            random_state=config.random_state,
            stratify=config.group_stratify,
        )
    else:
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state,
        )
        outer_splits = outer_cv.split(X_full, y_full)

    inner_cv_rows: List[Dict[str, Any]] = []
    inner_cv_split_rows: List[Dict[str, Any]] = []
    outer_rows: List[Dict[str, Any]] = []

    all_roc_data = {"fpr": [], "tpr": [], "auc": []}
    all_pr_data = {"recall": [], "precision": [], "ap": []}

    logger.info(f"Global multiclass labels: n_classes={len(classes)}")
    for fold, (tr_idx, va_idx) in enumerate(outer_splits, 1):
        logger.info(f"Fold {fold}/{config.outer_folds}")
        fold_root = config.outdir / f"fold{fold}"
        fold_root.mkdir(exist_ok=True)

        X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]
        groups_tr = None
        if df_groups is not None and groups_series is not None:
            assert_no_patient_leakage(
                df_groups,
                config.patient_col,
                np.asarray(tr_idx),
                np.asarray(va_idx),
                f"multiclass outer fold {fold}",
            )
            groups_tr = groups_series.iloc[tr_idx]
            n_patients_tr = groups_tr.nunique()
            n_patients_va = groups_series.iloc[va_idx].nunique()
            logger.info(
                f"  Patient split: train_patients={n_patients_tr}, val_patients={n_patients_va}, "
                f"train_rows={len(tr_idx)}, val_rows={len(va_idx)}"
            )

        _log_fold_class_coverage(y_tr, y_va, label_ids, classes, fold)

        for variant in variants:
            logger.info(f"  Variant: {variant}")
            var_dir = fold_root / f"multiclass_{variant}"
            var_dir.mkdir(exist_ok=True)

            best_model_name, best_estimator, roc_data, pr_data = _run_multiclass_variant(
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                classes=classes,
                label_ids=label_ids,
                estimators=estimators,
                param_grids=param_grids,
                scorers=scorers,
                variant=variant,
                fold=fold,
                config=config,
                inner_cv_rows=inner_cv_rows,
                inner_cv_split_rows=inner_cv_split_rows,
                outer_rows=outer_rows,
                var_dir=var_dir,
                groups_tr=groups_tr,
            )

            _save_multiclass_artifacts(
                var_dir=var_dir,
                model_name=best_model_name,
                estimator=best_estimator,
                classes=classes,
                feature_list=X_full.columns.tolist(),
            )

            if roc_data is not None:
                all_roc_data["fpr"].append(roc_data["fpr"])
                all_roc_data["tpr"].append(roc_data["tpr"])
                all_roc_data["auc"].append(roc_data["auc"])

            if pr_data is not None:
                all_pr_data["recall"].append(pr_data["recall"])
                all_pr_data["precision"].append(pr_data["precision"])
                all_pr_data["ap"].append(pr_data["ap"])

    _save_multiclass_results(
        outdir=config.outdir,
        inner_cv_rows=inner_cv_rows,
        inner_cv_split_rows=inner_cv_split_rows,
        outer_rows=outer_rows,
    )

    if len(all_roc_data["fpr"]) > 0:
        plot_averaged_roc_curves(
            all_roc_data["fpr"],
            all_roc_data["tpr"],
            all_roc_data["auc"],
            "Multiclass - ROC Curve (Averaged Across Folds)",
            config.outdir / "averaged_roc.png",
            show_individual=(config.outer_folds <= 5),
        )

        plot_averaged_pr_curves(
            all_pr_data["recall"],
            all_pr_data["precision"],
            all_pr_data["ap"],
            "Multiclass - PR Curve (Averaged Across Folds)",
            config.outdir / "averaged_pr.png",
            show_individual=(config.outer_folds <= 5),
        )

    return {
        "outdir": config.outdir,
        "n_folds": config.outer_folds,
        "variants": variants,
    }


def _run_multiclass_variant(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    classes: List[str],
    label_ids: List[int],
    estimators: Dict[str, Any],
    param_grids: Dict[str, Dict[str, list]],
    scorers: Dict[str, Any],
    variant: str,
    fold: int,
    config: MulticlassConfig,
    inner_cv_rows: List[Dict[str, Any]],
    inner_cv_split_rows: List[Dict[str, Any]],
    outer_rows: List[Dict[str, Any]],
    var_dir: Path,
    groups_tr: Optional[pd.Series] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run inner CV and evaluation for one SMOTE variant."""
    if config.patient_col and groups_tr is not None:
        patient_df = pd.DataFrame(
            {config.patient_col: np.asarray(groups_tr), "label": y_tr.values},
            index=X_tr.index,
        )
        patient_labels = make_group_labels(patient_df, config.patient_col, "label")
        min_class = int(pd.Series(patient_labels).value_counts().min())
    else:
        min_class = int(y_tr.value_counts().min())
    n_splits_eff = max(2, min(config.inner_splits, min_class))
    if n_splits_eff < config.inner_splits:
        logger.debug(f"Reducing inner_splits {config.inner_splits} -> {n_splits_eff} (minority={min_class})")

    if config.patient_col and groups_tr is not None:
        df_groups_tr = pd.DataFrame({config.patient_col: np.asarray(groups_tr)}, index=X_tr.index)
        inner_splits = list(iter_inner_splits(
            df_tr=df_groups_tr,
            y_tr=y_tr,
            patient_col=config.patient_col,
            n_splits=n_splits_eff,
            n_repeats=config.inner_repeats,
            random_state=config.random_state,
            stratify=config.group_stratify,
        ))
        for split_idx, (inner_tr_idx, inner_va_idx) in enumerate(inner_splits, 1):
            assert_no_patient_leakage(
                df_groups_tr,
                config.patient_col,
                np.asarray(inner_tr_idx),
                np.asarray(inner_va_idx),
                f"multiclass outer fold {fold} inner split {split_idx}",
            )
        cv_inner = inner_splits
        n_inner_total = len(inner_splits)
    else:
        cv_inner = RepeatedStratifiedKFold(
            n_splits=n_splits_eff,
            n_repeats=config.inner_repeats,
            random_state=config.random_state,
        )
        n_inner_total = n_splits_eff * config.inner_repeats

    sampler = AdaptiveSMOTE(k_max=5, random_state=config.random_state) if variant == "smote" else "passthrough"

    best_score = -np.inf
    best_model_name = None
    best_estimator = None

    for model_name, est in estimators.items():
        model_backend = "Torch" if model_name.startswith("torch_") else "Sklearn"
        logger.info("Training %s Model %s", model_backend, model_name)
        scaler = _make_scaler(X_tr)
        pipe = ImbPipeline([
            ("sampler", sampler),
            ("scaler", scaler),
            ("clf", est),
        ])

        grid = GridSearchCV(
            pipe,
            param_grids[model_name],
            cv=cv_inner,
            scoring=scorers,
            refit=REFIT_SCORER,
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
            error_score=np.nan,
        )

        try:
            grid.fit(X_tr, y_tr)
        except Exception as exc:
            logger.warning(f"Grid fit failed for {model_name}: {exc}")
            continue

        cvres = grid.cv_results_
        for i in range(len(cvres["params"])):
            row = {
                "fold": fold,
                "sampler": variant,
                "task": "multiclass",
                "model_name": model_name,
                "rank_test_f1_macro": int(cvres.get(f"rank_test_{REFIT_SCORER}", [np.nan] * len(cvres["params"]))[i]),
                "mean_test_f1_macro": float(cvres.get(f"mean_test_{REFIT_SCORER}", [np.nan] * len(cvres["params"]))[i]),
                "std_test_f1_macro": float(cvres.get(f"std_test_{REFIT_SCORER}", [np.nan] * len(cvres["params"]))[i]),
            }
            for k, v in cvres["params"][i].items():
                row[k.replace("clf__", "")] = v
            inner_cv_rows.append(row)

        best_idx = grid.best_index_
        tm_label = f"multiclass__{model_name} [{'SMOTE' if variant == 'smote' else 'No-SMOTE'}]"
        for s in range(n_inner_total):
            rec = {
                "task_model": tm_label,
                "outer_fold": fold,
                "inner_split": int(s + 1),
            }
            ok = True
            for name in MC_SCORER_ORDER:
                key = f"split{s}_test_{name}"
                if key not in cvres:
                    ok = False
                    break
                val = cvres[key][best_idx]
                rec[name] = float(val) if val == val else np.nan
            if ok:
                inner_cv_split_rows.append(rec)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model_name = model_name
            best_estimator = grid.best_estimator_

        model_train_metrics = _compute_multiclass_metrics(grid.best_estimator_, X_tr, y_tr, label_ids)
        model_val_metrics = _compute_multiclass_metrics(grid.best_estimator_, X_va, y_va, label_ids)
        outer_rows.append({
            "fold": fold,
            "sampler": variant,
            "phase": "train",
            "task": "multiclass",
            "model_name": model_name,
            **model_train_metrics,
        })
        outer_rows.append({
            "fold": fold,
            "sampler": variant,
            "phase": "val",
            "task": "multiclass",
            "model_name": model_name,
            **model_val_metrics,
        })

    if best_estimator is None:
        logger.warning(f"No model succeeded for fold {fold} variant {variant}")
        return None, None, None, None

    logger.info(f"    Best model: {best_model_name} (F1={best_score:.3f})")

    val_metrics, y_va_pred, y_va_proba = _compute_multiclass_metrics(
        best_estimator,
        X_va,
        y_va,
        label_ids,
        return_preds=True,
    )

    roc_data, pr_data = _plot_multiclass_outputs(
        y_va=y_va,
        y_va_pred=y_va_pred,
        y_va_proba=y_va_proba,
        classes=classes,
        fold=fold,
        variant=variant,
        var_dir=var_dir,
    )

    return best_model_name, best_estimator, roc_data, pr_data


def _compute_multiclass_metrics(
    estimator: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    label_ids: List[int],
    return_preds: bool = False,
) -> Any:
    """Compute multiclass metrics with optional predictions."""
    y_pred = estimator.predict(X)
    y_proba = _get_probabilities(estimator, X, label_ids)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": _balanced_accuracy_with_labels(y_true, y_pred, label_ids),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=label_ids, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = np.nan
    else:
        metrics["roc_auc_ovr_macro"] = np.nan

    if return_preds:
        return metrics, y_pred, y_proba
    return metrics


def _plot_multiclass_outputs(
    y_va: pd.Series,
    y_va_pred: np.ndarray,
    y_va_proba: Optional[np.ndarray],
    classes: List[str],
    fold: int,
    variant: str,
    var_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Generate per-fold plots and return ROC/PR data for averaging."""
    roc_data = None
    pr_data = None

    if y_va_proba is not None:
        try:
            plot_roc_curve(
                y_va.values,
                y_va_proba,
                classes,
                f"Multiclass - ROC Curve (Fold {fold}, {variant})",
                var_dir / f"roc_multiclass_fold{fold}.png",
            )

            plot_pr_curve(
                y_va.values,
                y_va_proba,
                classes,
                f"Multiclass - PR Curve (Fold {fold}, {variant})",
                var_dir / f"pr_multiclass_fold{fold}.png",
            )

            n_classes = len(classes)
            y_bin = label_binarize(y_va.values, classes=list(range(n_classes)))
            if y_bin.ndim == 1:
                y_bin = np.column_stack([1 - y_bin, y_bin])

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_va.values, y_va_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_va.values, y_va_proba[:, 1])
                ap = average_precision_score(y_va.values, y_va_proba[:, 1])
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_va_proba.ravel())
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_va_proba.ravel())
                ap = average_precision_score(y_bin, y_va_proba, average="micro")

            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
            pr_data = {"recall": recall, "precision": precision, "ap": ap}

        except Exception as exc:
            logger.warning(f"Failed to generate ROC/PR curves for fold {fold}: {exc}")

    try:
        plot_confusion_matrix(
            y_va.values,
            y_va_pred,
            classes,
            f"Multiclass - Confusion Matrix (Fold {fold}, {variant})",
            var_dir / f"confusion_matrix_fold{fold}.png",
            normalize="true",
        )
    except Exception as exc:
        logger.warning(f"Failed to generate confusion matrix for fold {fold}: {exc}")

    return roc_data, pr_data


def _balanced_accuracy_with_labels(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label_ids: List[int],
) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
    per_class = np.nan_to_num(per_class)
    return float(per_class.mean()) if per_class.size else float("nan")


def _get_multiclass_scorers(label_ids: List[int]) -> Dict[str, Any]:
    """Scorers for multiclass GridSearchCV."""
    return {
        "Accuracy": make_scorer(accuracy_score),
        "Balanced Accuracy": make_scorer(_balanced_accuracy_with_labels, label_ids=label_ids),
        "F1 Macro": make_scorer(f1_score, average="macro", labels=label_ids, zero_division=0),
        "F1 Weighted": make_scorer(f1_score, average="weighted", labels=label_ids, zero_division=0),
    }


def _make_scaler(X: pd.DataFrame) -> StandardScaler:
    """Create a scaler compatible with dense or sparse inputs."""
    with_mean = True
    try:
        from scipy import sparse  # type: ignore

        if sparse.issparse(X):
            with_mean = False
    except Exception:
        pass
    return StandardScaler(with_mean=with_mean)


def _log_fold_class_coverage(
    y_tr: pd.Series,
    y_va: pd.Series,
    label_ids: List[int],
    classes: List[str],
    fold: int,
) -> None:
    train_counts = pd.Series(y_tr).value_counts().reindex(label_ids, fill_value=0)
    val_counts = pd.Series(y_va).value_counts().reindex(label_ids, fill_value=0)
    train_summary = ", ".join(f"{classes[i]}={int(train_counts[i])}" for i in label_ids)
    val_summary = ", ".join(f"{classes[i]}={int(val_counts[i])}" for i in label_ids)
    logger.info(f"  Fold {fold} class coverage: train[{train_summary}] val[{val_summary}]")

    missing = [classes[i] for i in label_ids if val_counts[i] == 0]
    if missing:
        logger.warning(f"  Fold {fold} missing classes in val: {missing}")


def _get_probabilities(
    estimator: Any,
    X: pd.DataFrame,
    label_ids: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Return class probabilities or decision scores aligned to global labels."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return _align_probabilities(estimator, proba, label_ids)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([1 - scores, scores])
        return _align_probabilities(estimator, scores, label_ids)
    return None


def _align_probabilities(
    estimator: Any,
    proba: np.ndarray,
    label_ids: Optional[List[int]],
) -> np.ndarray:
    """Align probability columns to global label ordering."""
    if not label_ids or not hasattr(estimator, "classes_"):
        return proba

    classes = list(getattr(estimator, "classes_", []))
    if len(classes) == len(label_ids) and classes == label_ids:
        return proba

    n_samples = proba.shape[0]
    aligned = np.zeros((n_samples, len(label_ids)), dtype=proba.dtype)
    label_index = {label: idx for idx, label in enumerate(label_ids)}
    for src_idx, label in enumerate(classes):
        if label in label_index:
            aligned[:, label_index[label]] = proba[:, src_idx]
    return aligned


def _apply_device_to_estimators(estimators: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Apply device selection to estimators that support it."""
    updated = {}
    for name, est in estimators.items():
        updated[name] = _apply_device_to_estimator(est, device)
    return updated


def _filter_estimators(
    estimators: Dict[str, Any],
    param_grids: Dict[str, Dict[str, list]],
    mode: str,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, list]]]:
    """Filter estimators and param grids by selection mode."""
    if mode == "all":
        return estimators, param_grids

    if mode == "torch_only":
        names = [name for name in estimators if name.startswith("torch_")]
    elif mode == "cpu_only":
        names = [name for name in estimators if not name.startswith("torch_")]
    else:
        raise ValueError(f"Unsupported estimator_mode: {mode}")

    filtered_estimators = {name: estimators[name] for name in names}
    filtered_grids = {name: param_grids[name] for name in names if name in param_grids}

    if not filtered_estimators:
        raise ValueError(f"No estimators available after applying estimator_mode={mode}")

    logger.info("Estimator mode: %s", mode)
    return filtered_estimators, filtered_grids


def _log_torch_status(requested_device: str, resolved_device: str) -> None:
    """Log torch availability and MPS status for multiclass training."""
    try:
        import torch
    except Exception as exc:
        logger.info("Torch available: no (%s)", exc)
        return

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    logger.info("Torch available: yes (cuda=%s, mps=%s)", cuda_available, mps_available)
    if requested_device in {"mps", "cuda"} and resolved_device != requested_device:
        logger.warning("Requested device %s resolved to %s.", requested_device, resolved_device)


def _apply_device_to_estimator(estimator: Any, device: str) -> Any:
    """Set device on estimator if supported."""
    try:
        params = estimator.get_params()
    except Exception:
        params = {}

    if "device" in params:
        try:
            estimator.set_params(device=device)
        except Exception:
            pass
    elif hasattr(estimator, "device"):
        try:
            setattr(estimator, "device", device)
        except Exception:
            pass

    return estimator


def _save_multiclass_artifacts(
    var_dir: Path,
    model_name: Optional[str],
    estimator: Optional[Any],
    classes: List[str],
    feature_list: List[str],
) -> None:
    """Persist model and metadata for a fold/variant."""
    if estimator is None:
        return

    joblib.dump(estimator, var_dir / "multiclass_model.joblib")
    if model_name:
        (var_dir / "multiclass_model_name.txt").write_text(model_name)
    pd.Series(classes).to_csv(var_dir / "classes.csv", index=False, header=False)
    pd.Series(feature_list).to_csv(var_dir / "feature_list.csv", index=False, header=False)


def _save_multiclass_results(
    outdir: Path,
    inner_cv_rows: List[Dict[str, Any]],
    inner_cv_split_rows: List[Dict[str, Any]],
    outer_rows: List[Dict[str, Any]],
) -> None:
    """Save nested CV CSV outputs with compatibility aliases."""
    if inner_cv_rows:
        inner_df = pd.DataFrame(inner_cv_rows)
        inner_df.to_csv(outdir / "inner_cv_results.csv", index=False)
        inner_df.to_csv(outdir / "metrics_inner_cv.csv", index=False)

    if inner_cv_split_rows:
        split_df = pd.DataFrame(inner_cv_split_rows, columns=["task_model", "outer_fold", "inner_split"] + MC_SCORER_ORDER)
        split_df.to_csv(outdir / "inner_cv_splits.csv", index=False)
        split_df.to_csv(outdir / "metrics_inner_cv_splits.csv", index=False)

    if outer_rows:
        outer_df = pd.DataFrame(outer_rows)
        outer_df.to_csv(outdir / "outer_results.csv", index=False)
        outer_df.to_csv(outdir / "metrics_outer_multiclass_eval.csv", index=False)
