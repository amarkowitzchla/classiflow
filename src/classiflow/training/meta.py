"""Meta-classifier training for multiclass problems via binary task scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from classiflow.config import MetaConfig
from classiflow.io import load_data, load_data_with_groups, validate_data
from classiflow.tasks import TaskBuilder, load_composite_tasks
from classiflow.models import AdaptiveSMOTE
from classiflow.backends.registry import get_backend, get_model_set
from classiflow.metrics.scorers import get_scorers, SCORER_ORDER
from classiflow.metrics.binary import compute_binary_metrics
from classiflow.metrics.calibration import compute_probability_quality
from classiflow.metrics.decision import compute_decision_metrics
from classiflow.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
)
from classiflow.lineage.manifest import create_training_manifest
from classiflow.lineage.hashing import get_file_metadata
from classiflow.splitting import (
    iter_outer_splits,
    iter_inner_splits,
    assert_no_patient_leakage,
    make_group_labels,
)
from classiflow.tracking import get_tracker, extract_loggable_params, summarize_metrics

logger = logging.getLogger(__name__)


def _log_torch_status(requested_device: str) -> None:
    """Log torch availability for GPU-backed meta training."""
    try:
        import torch
    except Exception as exc:
        logger.info("Torch available: no (%s)", exc)
        return

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    logger.info("Torch available: yes (cuda=%s, mps=%s)", cuda_available, mps_available)
    logger.info("Requested device: %s", requested_device)


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
    data_path = config.resolved_data_path
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Label: {config.label_col}")
    logger.info(f"  SMOTE: {config.smote_mode}")
    logger.info(f"  Backend: {config.backend}")

    # Initialize experiment tracker
    tracker = get_tracker(
        backend=config.tracker,
        experiment_name=config.experiment_name or "classiflow-meta",
    )
    tracker.start_run(
        run_name=config.run_name,
        tags=config.tracker_tags,
    )

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # Load data
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

    # Filter to specified classes if provided
    if config.classes:
        mask = y_full.isin(config.classes)
        X_full = X_full[mask]
        y_full = y_full[mask]
        if groups is not None:
            groups = groups[mask]
        classes = config.classes
    else:
        classes = sorted(y_full.unique().tolist())

    logger.info(f"Classes: {classes}")
    validate_data(X_full, y_full)

    if config.patient_col and groups is not None:
        patient_df = pd.DataFrame(
            {config.patient_col: groups, "label": y_full.values},
            index=X_full.index,
        )
        make_group_labels(patient_df, config.patient_col, "label")

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
    file_metadata = get_file_metadata(data_path)

    # Build task definitions for manifest
    task_definitions = {name: str(func) for name, func in tasks.items()}

    config_dict = config.to_dict()
    config_dict["stratification_level"] = "patient" if config.patient_col else "sample"

    manifest = create_training_manifest(
        data_path=data_path,
        data_hash=file_metadata["sha256_hash"],
        data_size_bytes=file_metadata["size_bytes"],
        data_row_count=file_metadata.get("row_count"),
        config=config_dict,
        task_type="meta",
        feature_list=X_full.columns.tolist(),
        task_definitions=task_definitions,
        hierarchical=False,
    )
    manifest.save(config.outdir / "run.json")
    logger.info(f"Saved training manifest: run_id={manifest.run_id}")

    backend = get_backend(config.backend)
    if backend == "sklearn" and config.device != "auto":
        logger.info("  Device setting is ignored for sklearn backend.")
    if backend == "torch":
        _log_torch_status(config.device)
        try:
            import torch
        except Exception as exc:
            raise ValueError(f"Torch backend requested but torch is unavailable: {exc}") from exc
        if config.require_torch_device:
            if config.device == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS device requested but not available; set --device cpu or fix MPS setup.")
            if config.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available; set --device cpu or fix CUDA setup.")

    # Run nested CV with meta-classifier
    results = _run_meta_nested_cv(
        X_full=X_full,
        y_full=y_full,
        tasks=tasks,
        config=config,
        groups=groups,
    )

    # Log to experiment tracker
    tracker.log_params(extract_loggable_params(config))
    tracker.set_tags({
        "task_type": "meta",
        "backend": config.backend,
        "smote_mode": config.smote_mode,
        "run_id": manifest.run_id,
        "num_classes": str(len(config.classes)) if config.classes else "auto",
    })

    # Log summary metrics if available
    if "summary" in results:
        tracker.log_metrics(summarize_metrics(results["summary"]))

    # Log artifacts
    tracker.log_artifact(config.outdir / "run.json")
    for csv_file in config.outdir.glob("metrics_*.csv"):
        tracker.log_artifact(csv_file)
    for png_file in config.outdir.glob("*.png"):
        tracker.log_artifact(png_file)

    tracker.end_run()

    logger.info("Meta-classifier training complete")
    return results


def _run_meta_nested_cv(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    tasks: Dict[str, Callable],
    config: MetaConfig,
    groups: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Run nested CV for meta-classifier."""
    model_spec = get_model_set(
        command="train-meta",
        backend=get_backend(config.backend),
        model_set=config.model_set,
        random_state=config.random_state,
        max_iter=config.max_iter,
        device=config.device,
        torch_dtype=config.torch_dtype,
        torch_num_workers=config.torch_num_workers,
        meta_C_grid=config.meta_C_grid,
    )
    estimators = model_spec["base_estimators"]
    param_grids = model_spec["base_param_grids"]
    meta_estimators = model_spec["meta_estimators"]
    meta_param_grids = model_spec["meta_param_grids"]
    scorers = get_scorers()

    # SMOTE variants
    variants = ["smote", "none"] if config.smote_mode in ("on", "both") else ["none"]

    # Outer CV
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
        )
    else:
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state,
        )
        outer_splits = outer_cv.split(X_full, y_full)

    # Collectors
    inner_cv_rows = []
    inner_cv_split_rows = []
    outer_bin_rows = []
    outer_meta_rows = []
    calibration_comparison: Dict[str, Any] = {"folds": {}}

    # Collectors for averaged plots across folds
    all_roc_data = {"fpr": [], "tpr": [], "auc": []}
    all_pr_data = {"recall": [], "precision": [], "ap": []}

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
                f"meta outer fold {fold}",
            )
            groups_tr = groups_series.iloc[tr_idx]

        # Inner CV (per fold for meta classifier)
        if config.patient_col and groups_tr is not None:
            df_groups_tr = pd.DataFrame({config.patient_col: groups_tr}, index=X_tr.index)
            inner_splits = list(iter_inner_splits(
                df_tr=df_groups_tr,
                y_tr=y_tr,
                patient_col=config.patient_col,
                n_splits=config.inner_splits,
                n_repeats=config.inner_repeats,
                random_state=config.random_state,
            ))
            for split_idx, (inner_tr_idx, inner_va_idx) in enumerate(inner_splits, 1):
                assert_no_patient_leakage(
                    df_groups_tr,
                    config.patient_col,
                    np.asarray(inner_tr_idx),
                    np.asarray(inner_va_idx),
                    f"meta outer fold {fold} inner split {split_idx}",
                )
            cv_inner = inner_splits
            n_inner_total = len(inner_splits)
        else:
            cv_inner = RepeatedStratifiedKFold(
                n_splits=config.inner_splits,
                n_repeats=config.inner_repeats,
                random_state=config.random_state,
            )
            n_inner_total = config.inner_splits * config.inner_repeats

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
                groups_tr=groups_tr,
                patient_col=config.patient_col,
                inner_splits=config.inner_splits,
                inner_repeats=config.inner_repeats,
            )

            # Save binary artifacts
            var_dir = fold_root / f"binary_{variant}"
            var_dir.mkdir(exist_ok=True)
            joblib.dump({"pipes": best_pipes, "best_models": best_models}, var_dir / "binary_pipes.joblib")

            # Build meta-features and train meta-classifier
            roc_data, pr_data, fold_calibration, fold_confusion = _train_meta_model(
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
                groups_tr=groups_tr,
                meta_estimators=meta_estimators,
                meta_param_grids=meta_param_grids,
            )
            if fold_calibration:
                key = f"fold_{fold}_{variant}"
                fold_calibration["fold"] = fold
                fold_calibration["variant"] = variant
                calibration_comparison["folds"][key] = fold_calibration
            if fold_confusion:
                if "confusion_matrices" not in calibration_comparison:
                    calibration_comparison["confusion_matrices"] = {
                        "labels": fold_confusion["labels"],
                        "folds": {},
                    }
                calibration_comparison["confusion_matrices"]["folds"][key] = {
                    "matrix": fold_confusion["matrix"],
                    "variant": variant,
                    "fold": fold,
                }

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
    inner_split_df = pd.DataFrame(inner_cv_split_rows, columns=["task_model", "outer_fold", "inner_split"] + SCORER_ORDER)
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

    if calibration_comparison["folds"]:
        calibration_comparison["selection"] = _select_calibration_method(calibration_comparison["folds"])
        conf = calibration_comparison.get("confusion_matrices")
        if conf and conf.get("folds"):
            aggregate = None
            for entry in conf["folds"].values():
                matrix = np.array(entry["matrix"])
                aggregate = matrix if aggregate is None else aggregate + matrix
            if aggregate is not None:
                conf["aggregate"] = aggregate.tolist()
        _write_json(config.outdir / "calibration_comparison.json", calibration_comparison)

    return {
        "outdir": config.outdir,
        "n_tasks": len(tasks),
        "n_folds": config.outer_folds,
        "variants": variants,
    }


def _train_binary_tasks(
    X_tr,
    y_tr,
    X_va,
    y_va,
    tasks,
    estimators,
    param_grids,
    scorers,
    cv_inner,
    n_inner_total,
    variant,
    fold,
    random_state,
    inner_cv_rows,
    inner_cv_split_rows,
    outer_bin_rows,
    groups_tr: Optional[pd.Series] = None,
    patient_col: Optional[str] = None,
    inner_splits: int = 5,
    inner_repeats: int = 2,
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

        cv_for_task = cv_inner
        n_inner_total_task = n_inner_total
        if patient_col and groups_tr is not None:
            groups_task = groups_tr.loc[y_bin.index]
            df_groups_task = pd.DataFrame({patient_col: groups_task}, index=X_bin.index)
            inner_splits_task = list(iter_inner_splits(
                df_tr=df_groups_task,
                y_tr=y_bin,
                patient_col=patient_col,
                n_splits=inner_splits,
                n_repeats=inner_repeats,
                random_state=random_state,
            ))
            for split_idx, (inner_tr_idx, inner_va_idx) in enumerate(inner_splits_task, 1):
                assert_no_patient_leakage(
                    df_groups_task,
                    patient_col,
                    np.asarray(inner_tr_idx),
                    np.asarray(inner_va_idx),
                    f"{task_name} outer fold {fold} inner split {split_idx}",
                )
            cv_for_task = inner_splits_task
            n_inner_total_task = len(inner_splits_task)

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
                pipe,
                param_grids[model_name],
                cv=cv_for_task,
                scoring=scorers,
                refit="F1 Score",
                n_jobs=1,
                verbose=0,
                return_train_score=False,
                error_score=np.nan,
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
            for s in range(n_inner_total_task):
                rec = {
                    "task_model": tm_label,
                    "outer_fold": fold,
                    "inner_split": int(s + 1),
                }
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
    X_tr,
    y_tr,
    X_va,
    y_va,
    best_pipes,
    best_models,
    tasks,
    cv_inner,
    variant,
    fold,
    var_dir,
    config,
    outer_meta_rows,
    groups_tr: Optional[pd.Series],
    meta_estimators,
    meta_param_grids,
):
    """Train meta-classifier on binary scores."""
    X_meta_tr = _build_meta_features(
        X_tr,
        y_tr,
        best_pipes,
        best_models,
        tasks,
        config,
        groups_tr=groups_tr,
        use_oof=True,
        fold=fold,
        variant=variant,
    )
    X_meta_va = _build_meta_features(
        X_va,
        y_va,
        best_pipes,
        best_models,
        tasks,
        config,
        groups_tr=None,
        use_oof=False,
        fold=fold,
        variant=variant,
    )

    X_meta_tr, y_tr_meta, groups_tr_meta = _filter_meta_training_rows(
        X_meta_tr, y_tr, groups_tr, fold, variant
    )
    if X_meta_tr.empty or y_tr_meta.nunique() < 2:
        raise ValueError("Insufficient meta training samples after OOF filtering.")

    cv_inner_meta = cv_inner
    if config.patient_col and groups_tr_meta is not None:
        df_groups_meta = pd.DataFrame({config.patient_col: groups_tr_meta}, index=X_meta_tr.index)
        cv_inner_meta = list(iter_inner_splits(
            df_tr=df_groups_meta,
            y_tr=y_tr_meta,
            patient_col=config.patient_col,
            n_splits=config.inner_splits,
            n_repeats=config.inner_repeats,
            random_state=config.random_state,
        ))

    best_meta = None
    best_grid = None
    best_score = -np.inf
    best_name = None

    for model_name, meta in meta_estimators.items():
        grid = GridSearchCV(
            meta,
            meta_param_grids.get(model_name, {}),
            cv=cv_inner_meta,
            scoring="f1_macro",
            n_jobs=1,
            return_train_score=True,
        )
        grid.fit(X_meta_tr, y_tr_meta)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_meta = grid.best_estimator_
            best_grid = grid
            best_name = model_name

    if best_meta is None:
        raise ValueError("No meta estimator fit successfully.")

    calibrated_meta, calibration_metadata = _calibrate_meta_classifier(
        best_meta, X_meta_tr, y_tr_meta, config
    )

    joblib.dump(calibrated_meta, var_dir / "meta_model.joblib")
    pd.Series(list(X_meta_tr.columns)).to_csv(var_dir / "meta_features.csv", index=False, header=False)
    pd.Series(calibrated_meta.classes_).to_csv(var_dir / "meta_classes.csv", index=False, header=False)

    _write_json(var_dir / "calibration_metadata.json", calibration_metadata)

    threshold_config = {
        "strategy": "argmax",
        "binary_threshold": 0.5 if len(calibrated_meta.classes_) == 2 else None,
    }
    _write_json(var_dir / "threshold_config.json", threshold_config)

    y_pred_tr = calibrated_meta.predict(X_meta_tr)
    y_pred_va = calibrated_meta.predict(X_meta_va)
    y_prob_va = _safe_predict_proba(calibrated_meta, X_meta_va)

    y_pred_va_uncal = best_meta.predict(X_meta_va)
    y_prob_va_uncal = _safe_predict_proba(best_meta, X_meta_va)

    classes = [str(c) for c in calibrated_meta.classes_]

    cal_metrics, cal_curve = compute_probability_quality(
        y_true=y_va.tolist(),
        y_pred=y_pred_va.tolist(),
        y_proba=y_prob_va,
        classes=classes,
        bins=config.calibration_bins,
    )
    uncal_metrics, _ = compute_probability_quality(
        y_true=y_va.tolist(),
        y_pred=y_pred_va_uncal.tolist(),
        y_proba=y_prob_va_uncal,
        classes=classes,
        bins=config.calibration_bins,
    )

    calibration_summary = {
        "fold": fold,
        "variant": variant,
        "calibration_metadata": calibration_metadata,
        "calibrated": _serialize_metrics(cal_metrics),
        "uncalibrated": _serialize_metrics(uncal_metrics),
    }
    _write_json(var_dir / "calibration_summary.json", calibration_summary)
    cal_curve.to_csv(var_dir / "calibration_curve.csv", index=False)

    meta_train = {
        "fold": fold,
        "sampler": variant,
        "phase": "train",
        "model_name": best_name or "MetaModel",
        "accuracy": accuracy_score(y_tr_meta, y_pred_tr),
        "balanced_accuracy": balanced_accuracy_score(y_tr_meta, y_pred_tr),
        "f1_macro": f1_score(y_tr_meta, y_pred_tr, average="macro"),
        "f1_weighted": f1_score(y_tr_meta, y_pred_tr, average="weighted"),
        "calibration_method": calibration_metadata.get("method_used"),
        "calibration_enabled": calibration_metadata.get("enabled", False),
        "calibration_cv": calibration_metadata.get("cv"),
        "calibration_bins": config.calibration_bins,
    }

    decision_metrics = compute_decision_metrics(y_va.values, y_pred_va, classes)

    meta_val = {
        "fold": fold,
        "sampler": variant,
        "phase": "val",
        "model_name": best_name or "MetaModel",
        "accuracy": accuracy_score(y_va, y_pred_va),
        "balanced_accuracy": balanced_accuracy_score(y_va, y_pred_va),
        "sensitivity": decision_metrics.get("sensitivity"),
        "specificity": decision_metrics.get("specificity"),
        "ppv": decision_metrics.get("ppv"),
        "npv": decision_metrics.get("npv"),
        "recall": decision_metrics.get("sensitivity"),
        "precision": decision_metrics.get("ppv"),
        "f1_macro": f1_score(y_va, y_pred_va, average="macro"),
        "f1_weighted": f1_score(y_va, y_pred_va, average="weighted"),
        "meta_C": best_grid.best_params_.get("C") if best_grid else None,
        "brier": cal_metrics.get("brier"),
        "brier_calibrated": cal_metrics.get("brier"),
        "log_loss": cal_metrics.get("log_loss"),
        "log_loss_calibrated": cal_metrics.get("log_loss"),
        "ece": cal_metrics.get("ece"),
        "ece_calibrated": cal_metrics.get("ece"),
        "brier_uncalibrated": uncal_metrics.get("brier"),
        "log_loss_uncalibrated": uncal_metrics.get("log_loss"),
        "ece_uncalibrated": uncal_metrics.get("ece"),
        "calibration_method": calibration_metadata.get("method_used"),
        "calibration_enabled": calibration_metadata.get("enabled", False),
        "calibration_cv": calibration_metadata.get("cv"),
        "calibration_bins": config.calibration_bins,
        "calibration_warnings": "; ".join(calibration_metadata.get("warnings", [])),
    }

    outer_meta_rows.append(meta_train)
    outer_meta_rows.append(meta_val)

    fold_preds = pd.DataFrame(index=X_va.index)
    fold_preds["sample_id"] = fold_preds.index.astype(str)
    fold_preds["y_true"] = y_va.values
    fold_preds["y_pred"] = y_pred_va
    fold_preds["split"] = "outer_test"
    fold_preds["threshold_used"] = 0.5 if len(classes) == 2 else np.nan
    fold_preds["fold_id"] = fold
    fold_preds["variant"] = variant
    fold_preds["calibration_method"] = calibration_metadata.get("method_used")
    fold_preds["calibration_enabled"] = calibration_metadata.get("enabled", False)
    fold_preds["calibration_bins"] = config.calibration_bins
    fold_preds["calibration_cv"] = calibration_metadata.get("cv")
    fold_preds["y_prob"] = (
        np.max(y_prob_va, axis=1) if y_prob_va is not None else np.nan
    )
    fold_preds["y_score_raw"] = (
        np.max(y_prob_va_uncal, axis=1) if y_prob_va_uncal is not None else np.nan
    )
    if y_prob_va is not None:
        for i, cls in enumerate(classes):
            fold_preds[f"y_prob_{cls}"] = y_prob_va[:, i]
    if y_prob_va_uncal is not None:
        for i, cls in enumerate(classes):
            fold_preds[f"y_score_raw_{cls}"] = y_prob_va_uncal[:, i]
    fold_preds.to_csv(var_dir / "predictions_outer_test.csv", index=False)

    calibration_comparison = _compare_calibration_methods(
        model=best_meta,
        X_meta_tr=X_meta_tr,
        y_tr=y_tr_meta,
        X_meta_va=X_meta_va,
        y_va=y_va,
        classes=classes,
        config=config,
    )

    roc_data = None
    pr_data = None

    try:
        le = LabelEncoder()
        y_va_enc = le.fit_transform(y_va)
        y_pred_va_enc = le.transform(y_pred_va)
        encoded_classes = le.classes_.tolist()

        if y_prob_va is not None:
            plot_roc_curve(
                y_va_enc,
                y_prob_va,
                encoded_classes,
                f"Meta-Classifier – ROC Curve (Fold {fold}, {variant})",
                var_dir / f"roc_meta_fold{fold}.png",
            )

            plot_pr_curve(
                y_va_enc,
                y_prob_va,
                encoded_classes,
                f"Meta-Classifier – PR Curve (Fold {fold}, {variant})",
                var_dir / f"pr_meta_fold{fold}.png",
            )

        n_classes = len(encoded_classes)
        y_bin_va = label_binarize(y_va_enc, classes=range(n_classes))

        if y_prob_va is not None:
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_va_enc, y_prob_va[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
                precision, recall, _ = precision_recall_curve(y_va_enc, y_prob_va[:, 1])
                ap = average_precision_score(y_va_enc, y_prob_va[:, 1])
                pr_data = {"recall": recall, "precision": precision, "ap": ap}
            else:
                fpr_micro, tpr_micro, _ = roc_curve(y_bin_va.ravel(), y_prob_va.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                roc_data = {"fpr": fpr_micro, "tpr": tpr_micro, "auc": roc_auc_micro}
                precision_micro, recall_micro, _ = precision_recall_curve(
                    y_bin_va.ravel(), y_prob_va.ravel()
                )
                ap_micro = average_precision_score(y_bin_va.ravel(), y_prob_va.ravel())
                pr_data = {"recall": recall_micro, "precision": precision_micro, "ap": ap_micro}

        plot_confusion_matrix(
            y_va_enc,
            y_pred_va_enc,
            encoded_classes,
            f"Meta-Classifier – Confusion Matrix (Fold {fold}, {variant})",
            var_dir / f"cm_meta_fold{fold}.png",
            normalize="true",
        )

        logger.debug(f"Generated meta-classifier plots for fold {fold} in {var_dir}")

    except Exception as exc:
        logger.warning(f"Failed to generate plots for fold {fold}: {exc}")

    cm = confusion_matrix(y_va, y_pred_va, labels=classes)
    confusion_payload = {
        "labels": [str(c) for c in classes],
        "matrix": cm.tolist(),
    }

    return roc_data, pr_data, calibration_comparison, confusion_payload


def _compare_calibration_methods(
    model,
    X_meta_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_meta_va: pd.DataFrame,
    y_va: pd.Series,
    classes: list[str],
    config: MetaConfig,
) -> Dict[str, Any]:
    """Compare sigmoid vs isotonic calibration on a single fold."""
    comparison: Dict[str, Any] = {}
    for method in ("sigmoid", "isotonic"):
        calibrator, metadata = _fit_meta_calibrator(
            model=model,
            X_meta=X_meta_tr,
            y=y_tr,
            config=config,
            method=method,
            allow_isotonic_fallback=False,
        )
        entry = {
            "enabled": metadata.get("enabled", False),
            "warnings": metadata.get("warnings", []),
        }
        if metadata.get("enabled"):
            y_pred = calibrator.predict(X_meta_va)
            y_proba = _safe_predict_proba(calibrator, X_meta_va)
            if y_proba is not None:
                metrics, _ = compute_probability_quality(
                    y_true=y_va.tolist(),
                    y_pred=y_pred.tolist(),
                    y_proba=y_proba,
                    classes=classes,
                    bins=config.calibration_bins,
                )
                entry.update(_serialize_metrics(metrics))
            else:
                entry.update({"brier": None, "ece": None, "log_loss": None})
        else:
            entry.update({"brier": None, "ece": None, "log_loss": None})
        comparison[method] = entry
    return comparison


def _select_calibration_method(folds: Dict[str, Any]) -> Dict[str, Any]:
    """Select calibration method based on mean Brier and ECE."""
    metrics: Dict[str, Dict[str, list[float]]] = {
        "sigmoid": {"brier": [], "ece": [], "log_loss": []},
        "isotonic": {"brier": [], "ece": [], "log_loss": []},
    }
    for entry in folds.values():
        for method in ("sigmoid", "isotonic"):
            data = entry.get(method, {})
            if not data or not data.get("enabled"):
                continue
            for key in ("brier", "ece", "log_loss"):
                value = data.get(key)
                if value is not None:
                    metrics[method][key].append(float(value))

    summary: Dict[str, Any] = {}
    for method, values in metrics.items():
        summary[method] = {
            "n": len(values["brier"]),
            "brier_mean": float(np.mean(values["brier"])) if values["brier"] else None,
            "ece_mean": float(np.mean(values["ece"])) if values["ece"] else None,
            "log_loss_mean": float(np.mean(values["log_loss"])) if values["log_loss"] else None,
        }

    selected = "sigmoid"
    reason = "Default to sigmoid calibration."
    sig_brier = summary["sigmoid"]["brier_mean"]
    iso_brier = summary["isotonic"]["brier_mean"]
    sig_ece = summary["sigmoid"]["ece_mean"]
    iso_ece = summary["isotonic"]["ece_mean"]
    if sig_brier is not None and iso_brier is not None and sig_brier > 0:
        improvement = (sig_brier - iso_brier) / sig_brier
        if sig_ece is not None and iso_ece is not None and improvement >= 0.05 and iso_ece <= sig_ece:
            selected = "isotonic"
            reason = "Isotonic selected (>=5% mean Brier improvement without worse ECE)."

    return {
        "method_default": "sigmoid",
        "method_selected": selected,
        "reason": reason,
        "summary": summary,
    }


def _calibrate_meta_classifier(model, X_meta: pd.DataFrame, y: pd.Series, config: MetaConfig):
    """Apply optional probability calibration to the meta-classifier."""
    calibrator, metadata = _fit_meta_calibrator(
        model=model,
        X_meta=X_meta,
        y=y,
        config=config,
        method=config.calibration_method,
        allow_isotonic_fallback=True,
    )
    return calibrator, metadata


def _fit_meta_calibrator(
    model,
    X_meta: pd.DataFrame,
    y: pd.Series,
    config: MetaConfig,
    method: str,
    allow_isotonic_fallback: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit a calibrated classifier with explicit method controls."""
    metadata = {
        "enabled": False,
        "method_requested": method,
        "method_used": None,
        "cv": None,
        "bins": config.calibration_bins,
        "warnings": [],
    }

    if not config.calibrate_meta:
        metadata["warnings"].append("Calibration disabled in configuration.")
        return model, metadata

    if len(X_meta) < 2:
        metadata["warnings"].append("Insufficient samples for calibration.")
        return model, metadata

    effective_method = method
    y_series = pd.Series(y)
    if method == "isotonic":
        min_samples = config.calibration_isotonic_min_samples
        if len(X_meta) < min_samples or y_series.value_counts().min() < 2:
            if allow_isotonic_fallback:
                metadata["warnings"].append(
                    "Isotonic calibration not supported (min samples or class counts too low); falling back to sigmoid."
                )
                effective_method = "sigmoid"
            else:
                metadata["warnings"].append(
                    "Isotonic calibration not supported (min samples or class counts too low); skipping."
                )
                return model, metadata

    cv = max(2, min(config.calibration_cv, max(2, len(X_meta) - 1)))

    try:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=effective_method,
            cv=cv,
        )
        calibrator.fit(X_meta, y)
        metadata.update({
            "enabled": True,
            "method_used": effective_method,
            "cv": cv,
        })
        return calibrator, metadata
    except Exception as exc:
        metadata["warnings"].append(f"Calibration failed: {exc}")
        metadata.update({
            "enabled": False,
            "method_used": effective_method,
            "cv": cv,
        })
        return model, metadata


def _build_meta_features(
    X,
    y,
    best_pipes,
    best_models,
    tasks,
    config,
    groups_tr: Optional[pd.Series] = None,
    use_oof: bool = False,
    fold: Optional[int] = None,
    variant: Optional[str] = None,
):
    """Build meta-features from binary task scores."""
    meta = pd.DataFrame(index=X.index)

    for task_name, model_name in best_models.items():
        key = f"{task_name}__{model_name}"
        if key not in best_pipes:
            continue

        pipe = best_pipes[key]
        y_bin = tasks[task_name](y).dropna()
        if y_bin.empty:
            continue

        idx = y_bin.index
        X_subset = X.loc[idx]

        groups_subset = None
        if groups_tr is not None:
            groups_subset = groups_tr.loc[idx]

        if use_oof:
            splits = _inner_cv_splits_for_task(X_subset, y_bin, config, groups_subset)
            if splits:
                scores = _cross_val_scores(pipe, X_subset, y_bin, splits)
            else:
                scores = _cross_val_scores(pipe, X_subset, y_bin, [])
        else:
            scores = _get_scores(pipe, X_subset)

        col_name = f"{task_name}_score"
        meta[col_name] = 0.0
        meta.loc[idx, col_name] = scores

        if use_oof and np.isnan(scores).any():
            missing = int(np.isnan(scores).sum())
            logger.warning(
                "OOF scores missing for task=%s fold=%s variant=%s (n=%s).",
                task_name,
                fold if fold is not None else "unknown",
                variant if variant is not None else "unknown",
                missing,
            )

    return meta


def _filter_meta_training_rows(
    X_meta_tr: pd.DataFrame,
    y_tr: pd.Series,
    groups_tr: Optional[pd.Series],
    fold: int,
    variant: str,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Drop rows with missing OOF scores to avoid in-sample leakage."""
    nan_mask = X_meta_tr.isna().any(axis=1)
    if nan_mask.any():
        missing = int(nan_mask.sum())
        logger.warning(
            "Dropping %s meta rows with missing OOF scores (fold=%s variant=%s).",
            missing,
            fold,
            variant,
        )
        X_meta_tr = X_meta_tr.loc[~nan_mask]
        y_tr = y_tr.loc[X_meta_tr.index]
        if groups_tr is not None:
            groups_tr = groups_tr.loc[X_meta_tr.index]
    return X_meta_tr, y_tr, groups_tr


def _inner_cv_splits_for_task(
    X_task: pd.DataFrame,
    y_bin: pd.Series,
    config: MetaConfig,
    groups: Optional[pd.Series],
) -> list:
    """Return inner CV splits for meta-features (patient-aware when needed)."""
    if len(X_task) < 2:
        return []

    n_splits = max(2, min(config.inner_splits, max(2, len(X_task))))

    if config.patient_col and groups is not None and not groups.empty:
        df_groups_task = pd.DataFrame({config.patient_col: groups.values}, index=X_task.index)
        return list(iter_inner_splits(
            df_tr=df_groups_task,
            y_tr=y_bin,
            patient_col=config.patient_col,
            n_splits=n_splits,
            n_repeats=config.inner_repeats,
            random_state=config.random_state,
        ))

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=config.inner_repeats,
        random_state=config.random_state,
    )
    return list(splitter.split(X_task, y_bin))


def _cross_val_scores(pipe, X, y, splits):
    """Run cross-validated predictions to build out-of-fold scores."""
    if not splits:
        return np.full(len(X), np.nan)

    scores = pd.Series(np.nan, index=X.index)
    for train_idx, val_idx in splits:
        try:
            pipe_clone = clone(pipe)
            pipe_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.iloc[val_idx] = _get_scores(pipe_clone, X.iloc[val_idx])
        except Exception as exc:
            logger.warning(f"OOF prediction failed for split: {exc}")

    missing = scores.isna()
    if missing.any():
        scores.loc[missing] = np.nan

    return scores.values


def _safe_predict_proba(model, X):
    """Try to call predict_proba, return None on failure."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception as exc:
            logger.warning(f"predict_proba failed: {exc}")
    return None


def _serialize_metrics(metrics):
    """Convert NaN floats to None for JSON serialization."""
    serialized: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if value is None:
            serialized[key] = None
            continue
        try:
            if isinstance(value, float) and np.isnan(value):
                serialized[key] = None
            else:
                serialized[key] = float(value)
        except Exception:
            serialized[key] = value
    return serialized


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with directories created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _get_scores(pipe, X):
    """Extract scores from pipeline."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)
