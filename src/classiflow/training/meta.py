"""Meta-classifier training for multiclass problems via binary task scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Callable
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from classiflow.config import MetaConfig
from classiflow.io import load_data, validate_data
from classiflow.tasks import TaskBuilder, load_composite_tasks
from classiflow.models import get_estimators, get_param_grids, AdaptiveSMOTE
from classiflow.metrics.scorers import get_scorers, SCORER_ORDER
from classiflow.metrics.binary import compute_binary_metrics
from classiflow.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
)
from classiflow.lineage.manifest import create_training_manifest
from classiflow.lineage.hashing import get_file_metadata

logger = logging.getLogger(__name__)


def train_meta_classifier(config: MetaConfig) -> Dict[str, Any]:
    """
    Train meta-classifier pipeline for multiclass classification.

    Workflow:
    1. Load data and build binary tasks (OvR + pairwise + optional composite)
    2. For each outer fold:
       - Train binary models with inner CV
       - Build meta-features from binary scores
       - Train meta-classifier (multinomial logistic regression)
    3. Save artifacts and metrics

    Parameters
    ----------
    config : MetaConfig
        Meta-classifier training configuration

    Returns
    -------
    results : Dict[str, Any]
        Training results
    """
    logger.info("Starting meta-classifier training")
    logger.info(f"  Data: {config.data_csv}")
    logger.info(f"  Label: {config.label_col}")
    logger.info(f"  SMOTE: {config.smote_mode}")

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_full, y_full = load_data(config.data_csv, config.label_col, feature_cols=config.feature_cols)

    # Filter to specified classes if provided
    if config.classes:
        mask = y_full.isin(config.classes)
        X_full = X_full[mask]
        y_full = y_full[mask]
        classes = config.classes
    else:
        classes = sorted(y_full.unique().tolist())

    logger.info(f"Classes: {classes}")
    validate_data(X_full, y_full)

    # Build tasks
    task_builder = TaskBuilder(classes)

    if config.tasks_json and config.tasks_only:
        # Only load tasks from JSON, skip auto OvR/pairwise
        logger.info(f"Loading ONLY tasks from {config.tasks_json} (tasks_only=True)")
        task_builder = load_composite_tasks(config.tasks_json, task_builder)
    elif config.tasks_json:
        # Load auto tasks + tasks from JSON
        logger.info(f"Building auto OvR/pairwise tasks + tasks from {config.tasks_json}")
        task_builder = task_builder.build_all_auto_tasks()
        task_builder = load_composite_tasks(config.tasks_json, task_builder)
    else:
        # Only auto tasks
        logger.info("Building auto OvR/pairwise tasks")
        task_builder = task_builder.build_all_auto_tasks()

    tasks = task_builder.get_tasks()
    logger.info(f"Built {len(tasks)} tasks")

    # Create and save training manifest with lineage
    data_path = Path(config.data_csv)
    file_metadata = get_file_metadata(data_path)

    # Build task definitions for manifest
    task_definitions = {name: str(func) for name, func in tasks.items()}

    manifest = create_training_manifest(
        data_path=data_path,
        data_hash=file_metadata["sha256_hash"],
        data_size_bytes=file_metadata["size_bytes"],
        data_row_count=file_metadata.get("row_count"),
        config=config.to_dict(),
        task_type="meta",
        feature_list=X_full.columns.tolist(),
        task_definitions=task_definitions,
        hierarchical=False,
    )
    manifest.save(config.outdir / "run.json")
    logger.info(f"Saved training manifest: run_id={manifest.run_id}")

    # Run nested CV with meta-classifier
    results = _run_meta_nested_cv(
        X_full=X_full,
        y_full=y_full,
        tasks=tasks,
        config=config,
    )

    logger.info("Meta-classifier training complete")
    return results


def _run_meta_nested_cv(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    tasks: Dict[str, Callable],
    config: MetaConfig,
) -> Dict[str, Any]:
    """Run nested CV for meta-classifier."""
    estimators = get_estimators(config.random_state, config.max_iter)
    param_grids = get_param_grids()
    scorers = get_scorers()

    # SMOTE variants
    variants = ["smote", "none"] if config.smote_mode in ("on", "both") else ["none"]

    # Inner CV
    cv_inner = RepeatedStratifiedKFold(
        n_splits=config.inner_splits,
        n_repeats=config.inner_repeats,
        random_state=config.random_state,
    )
    n_inner_total = config.inner_splits * config.inner_repeats

    # Outer CV
    outer_cv = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    # Collectors
    inner_cv_rows = []
    inner_cv_split_rows = []
    outer_bin_rows = []
    outer_meta_rows = []

    # Collectors for averaged plots across folds
    all_roc_data = {"fpr": [], "tpr": [], "auc": []}
    all_pr_data = {"recall": [], "precision": [], "ap": []}

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_full, y_full), 1):
        logger.info(f"Fold {fold}/{config.outer_folds}")
        fold_root = config.outdir / f"fold{fold}"
        fold_root.mkdir(exist_ok=True)

        X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

        for variant in variants:
            logger.info(f"  Variant: {variant}")

            # Train binary tasks
            best_pipes, best_models = _train_binary_tasks(
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                tasks=tasks,
                estimators=estimators,
                param_grids=param_grids,
                scorers=scorers,
                cv_inner=cv_inner,
                n_inner_total=n_inner_total,
                variant=variant,
                fold=fold,
                random_state=config.random_state,
                inner_cv_rows=inner_cv_rows,
                inner_cv_split_rows=inner_cv_split_rows,
                outer_bin_rows=outer_bin_rows,
            )

            # Save binary artifacts
            var_dir = fold_root / f"binary_{variant}"
            var_dir.mkdir(exist_ok=True)
            joblib.dump({"pipes": best_pipes, "best_models": best_models}, var_dir / "binary_pipes.joblib")

            # Build meta-features and train meta-classifier
            roc_data, pr_data = _train_meta_model(
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                best_pipes=best_pipes,
                best_models=best_models,
                tasks=tasks,
                cv_inner=cv_inner,
                variant=variant,
                fold=fold,
                var_dir=var_dir,
                config=config,
                outer_meta_rows=outer_meta_rows,
            )

            # Collect ROC/PR data for averaged plots (use first variant only)
            if variant == variants[0] and roc_data is not None:
                all_roc_data["fpr"].append(roc_data["fpr"])
                all_roc_data["tpr"].append(roc_data["tpr"])
                all_roc_data["auc"].append(roc_data["auc"])

                all_pr_data["recall"].append(pr_data["recall"])
                all_pr_data["precision"].append(pr_data["precision"])
                all_pr_data["ap"].append(pr_data["ap"])

    # Save metrics
    pd.DataFrame(inner_cv_rows).to_csv(config.outdir / "metrics_inner_cv.csv", index=False)
    pd.DataFrame(outer_bin_rows).to_csv(config.outdir / "metrics_outer_binary_eval.csv", index=False)
    pd.DataFrame(outer_meta_rows).to_csv(config.outdir / "metrics_outer_meta_eval.csv", index=False)

    # Inner CV splits
    inner_split_df = pd.DataFrame(inner_cv_split_rows, columns=["task_model", "fold"] + SCORER_ORDER)
    inner_split_df.to_csv(config.outdir / "metrics_inner_cv_splits.csv", index=False)

    try:
        with pd.ExcelWriter(config.outdir / "metrics_inner_cv_splits.xlsx", engine="xlsxwriter") as writer:
            inner_split_df.to_excel(writer, index=False, sheet_name="InnerCV_Splits")
    except Exception as e:
        logger.warning(f"Could not write Excel file: {e}")

    # Generate averaged plots across folds
    if len(all_roc_data["fpr"]) > 0:
        logger.info("Generating averaged ROC and PR curves across folds")

        plot_averaged_roc_curves(
            all_roc_data["fpr"],
            all_roc_data["tpr"],
            all_roc_data["auc"],
            "Meta-Classifier – ROC Curve (Averaged Across Folds)",
            config.outdir / "roc_meta_averaged.png",
            show_individual=(config.outer_folds <= 5),
        )

        plot_averaged_pr_curves(
            all_pr_data["recall"],
            all_pr_data["precision"],
            all_pr_data["ap"],
            "Meta-Classifier – PR Curve (Averaged Across Folds)",
            config.outdir / "pr_meta_averaged.png",
            show_individual=(config.outer_folds <= 5),
        )

    logger.info(f"Saved metrics to {config.outdir}")

    return {
        "outdir": config.outdir,
        "n_tasks": len(tasks),
        "n_folds": config.outer_folds,
        "variants": variants,
    }


def _train_binary_tasks(
    X_tr, y_tr, X_va, y_va, tasks, estimators, param_grids, scorers,
    cv_inner, n_inner_total, variant, fold, random_state,
    inner_cv_rows, inner_cv_split_rows, outer_bin_rows,
):
    """Train binary models for all tasks."""
    best_pipes = {}
    best_models = {}

    sampler = AdaptiveSMOTE(k_max=5, random_state=random_state) if variant == "smote" else "passthrough"

    for task_name, labeler in tasks.items():
        y_bin = labeler(y_tr).dropna()
        if y_bin.nunique() < 2:
            continue

        X_bin = X_tr.loc[y_bin.index]

        best_f1, best_name, best_grid = -np.inf, None, None

        for model_name, est in estimators.items():
            # Build pipeline without VarianceThreshold to avoid removing all features
            # in small CV splits (especially problematic with scaled data)
            pipe = ImbPipeline([
                ("sampler", sampler),
                ("scaler", StandardScaler()),
                ("clf", est),
            ])

            grid = GridSearchCV(
                pipe, param_grids[model_name],
                cv=cv_inner, scoring=scorers, refit="F1 Score",
                n_jobs=-1, verbose=0, return_train_score=False, error_score=np.nan,
            )

            try:
                grid.fit(X_bin, y_bin)
            except Exception as e:
                logger.warning(f"Grid fit failed for {task_name}/{model_name}: {e}")
                continue

            # Log inner CV
            cvres = grid.cv_results_
            for i in range(len(cvres["params"])):
                row = {
                    "fold": fold,
                    "sampler": variant,
                    "task": task_name,
                    "model_name": model_name,
                    "rank_test_score": int(cvres.get("rank_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                    "mean_test_score": float(cvres.get("mean_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                    "std_test_score": float(cvres.get("std_test_F1 Score", [np.nan]*len(cvres["params"]))[i]),
                }
                for k, v in cvres["params"][i].items():
                    row[k.replace("clf__", "")] = v
                inner_cv_rows.append(row)

            # Per-split metrics
            best_idx = grid.best_index_
            tm_label = f"{task_name}__{model_name} [{'SMOTE' if variant=='smote' else 'No-SMOTE'}]"
            for s in range(n_inner_total):
                rec = {"task_model": tm_label, "fold": int(s + 1)}
                ok = True
                for name in SCORER_ORDER:
                    key = f"split{s}_test_{name}"
                    if key not in cvres:
                        ok = False
                        break
                    val = cvres[key][best_idx]
                    rec[name] = float(val) if val == val else np.nan
                if ok:
                    inner_cv_split_rows.append(rec)

            if grid.best_score_ > best_f1:
                best_f1, best_name, best_grid = grid.best_score_, model_name, grid

            best_pipes[f"{task_name}__{model_name}"] = grid.best_estimator_

        if best_name:
            best_models[task_name] = best_name
            logger.info(f"    {task_name}: {best_name} (F1={best_f1:.3f})")

            # Evaluate on train/val
            sel = best_pipes[f"{task_name}__{best_name}"]
            train_scores = _get_scores(sel, X_bin)
            train_metrics = compute_binary_metrics(y_bin.values.astype(int), train_scores)

            y_va_bin = labeler(y_va).dropna()
            if y_va_bin.nunique() >= 2:
                X_va_task = X_va.loc[y_va_bin.index]
                val_scores = _get_scores(sel, X_va_task)
                val_metrics = compute_binary_metrics(y_va_bin.values.astype(int), val_scores)
            else:
                val_metrics = {k: np.nan for k in train_metrics.keys()}

            outer_bin_rows.append({
                "fold": fold, "sampler": variant, "phase": "train",
                "task": task_name, "model_name": best_name, **train_metrics
            })
            outer_bin_rows.append({
                "fold": fold, "sampler": variant, "phase": "val",
                "task": task_name, "model_name": best_name, **val_metrics
            })

    return best_pipes, best_models


def _train_meta_model(
    X_tr, y_tr, X_va, y_va, best_pipes, best_models, tasks, cv_inner,
    variant, fold, var_dir, config, outer_meta_rows,
):
    """Train meta-classifier on binary scores."""
    # Build meta-features
    X_meta_tr = _build_meta_features(X_tr, y_tr, best_pipes, best_models, tasks)
    X_meta_va = _build_meta_features(X_va, y_va, best_pipes, best_models, tasks)

    # Train meta-classifier
    # Note: multi_class='multinomial' is now the default in sklearn 1.5+
    meta = LogisticRegression(
        class_weight="balanced",
        max_iter=config.max_iter,
        random_state=config.random_state,
    )

    grid = GridSearchCV(
        meta,
        {"C": config.meta_C_grid},
        cv=cv_inner,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=True,
    )
    grid.fit(X_meta_tr, y_tr)
    best_meta = grid.best_estimator_

    # Save meta artifacts
    joblib.dump(best_meta, var_dir / "meta_model.joblib")
    pd.Series(list(X_meta_tr.columns)).to_csv(var_dir / "meta_features.csv", index=False, header=False)
    pd.Series(best_meta.classes_).to_csv(var_dir / "meta_classes.csv", index=False, header=False)

    # Evaluate
    y_pred_tr = best_meta.predict(X_meta_tr)
    y_pred_va = best_meta.predict(X_meta_va)
    y_prob_va = best_meta.predict_proba(X_meta_va)

    meta_train = {
        "fold": fold, "sampler": variant, "phase": "train",
        "model_name": "MultinomialLogReg",
        "accuracy": accuracy_score(y_tr, y_pred_tr),
        "balanced_accuracy": balanced_accuracy_score(y_tr, y_pred_tr),
        "f1_macro": f1_score(y_tr, y_pred_tr, average="macro"),
        "f1_weighted": f1_score(y_tr, y_pred_tr, average="weighted"),
    }

    meta_val = {
        "fold": fold, "sampler": variant, "phase": "val",
        "model_name": "MultinomialLogReg",
        "accuracy": accuracy_score(y_va, y_pred_va),
        "balanced_accuracy": balanced_accuracy_score(y_va, y_pred_va),
        "f1_macro": f1_score(y_va, y_pred_va, average="macro"),
        "f1_weighted": f1_score(y_va, y_pred_va, average="weighted"),
        "meta_C": grid.best_params_.get("C"),
    }

    # Try ROC AUC
    roc_data = None
    pr_data = None

    try:
        y_bin_va = label_binarize(y_va, classes=best_meta.classes_)
        aucs_va = []
        for i in range(y_bin_va.shape[1]):
            if len(np.unique(y_bin_va[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin_va[:, i], y_prob_va[:, i])
            aucs_va.append(auc(fpr, tpr))
        meta_val["roc_auc_ovr_macro"] = float(np.mean(aucs_va)) if aucs_va else np.nan
    except Exception:
        meta_val["roc_auc_ovr_macro"] = np.nan

    outer_meta_rows.append(meta_train)
    outer_meta_rows.append(meta_val)

    # Generate per-fold plots
    try:
        # Encode labels for plotting
        le = LabelEncoder()
        y_va_enc = le.fit_transform(y_va)
        y_pred_va_enc = le.transform(y_pred_va)
        classes = le.classes_.tolist()

        # ROC curve
        plot_roc_curve(
            y_va_enc,
            y_prob_va,
            classes,
            f"Meta-Classifier – ROC Curve (Fold {fold}, {variant})",
            var_dir / f"roc_meta_fold{fold}.png",
        )

        # Collect ROC data for averaged plots (micro-average for multiclass)
        n_classes = len(classes)
        y_bin_va = label_binarize(y_va_enc, classes=range(n_classes))

        # Handle binary vs multiclass case
        if n_classes == 2:
            # Binary case: use positive class probabilities
            fpr, tpr, _ = roc_curve(y_va_enc, y_prob_va[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        else:
            # Multiclass: compute micro-average ROC curve
            fpr_micro, tpr_micro, _ = roc_curve(y_bin_va.ravel(), y_prob_va.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            roc_data = {"fpr": fpr_micro, "tpr": tpr_micro, "auc": roc_auc_micro}

        # PR curve
        plot_pr_curve(
            y_va_enc,
            y_prob_va,
            classes,
            f"Meta-Classifier – PR Curve (Fold {fold}, {variant})",
            var_dir / f"pr_meta_fold{fold}.png",
        )

        # Compute micro-average PR curve
        if n_classes == 2:
            # Binary case
            precision, recall, _ = precision_recall_curve(y_va_enc, y_prob_va[:, 1])
            ap = average_precision_score(y_va_enc, y_prob_va[:, 1])
            pr_data = {"recall": recall, "precision": precision, "ap": ap}
        else:
            # Multiclass: micro-average
            precision_micro, recall_micro, _ = precision_recall_curve(
                y_bin_va.ravel(), y_prob_va.ravel()
            )
            ap_micro = average_precision_score(y_bin_va.ravel(), y_prob_va.ravel())
            pr_data = {"recall": recall_micro, "precision": precision_micro, "ap": ap_micro}

        # Confusion matrix (use encoded labels)
        plot_confusion_matrix(
            y_va_enc,
            y_pred_va_enc,
            classes,
            f"Meta-Classifier – Confusion Matrix (Fold {fold}, {variant})",
            var_dir / f"cm_meta_fold{fold}.png",
            normalize="true",
        )

        logger.debug(f"Generated meta-classifier plots for fold {fold} in {var_dir}")

    except Exception as e:
        logger.warning(f"Failed to generate plots for fold {fold}: {e}")

    return roc_data, pr_data


def _build_meta_features(X, y, best_pipes, best_models, tasks):
    """Build meta-features from binary task scores."""
    meta = pd.DataFrame(index=X.index)
    for task_name, model_name in best_models.items():
        pipe = best_pipes[f"{task_name}__{model_name}"]
        y_bin = tasks[task_name](y).dropna()
        idx = y_bin.index
        scores = _get_scores(pipe, X.loc[idx])
        meta.loc[idx, f"{task_name}_score"] = scores
    return meta.fillna(0.0)


def _get_scores(pipe, X):
    """Extract scores from pipeline."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)
