"""Orchestration for project workflows."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import joblib

from classiflow import __version__
from classiflow.config import TrainConfig, MetaConfig, HierarchicalConfig, MulticlassConfig
from classiflow.evaluation.smote_comparison import SMOTEComparison
from classiflow.inference import InferenceConfig, run_inference
from classiflow.lineage.manifest import TrainingRunManifest
from classiflow.lineage.hashing import compute_file_hash
from classiflow.models import get_estimators, AdaptiveSMOTE
from classiflow.backends.registry import get_backend, get_model_set
from classiflow.tasks import TaskBuilder
from classiflow.training import (
    train_binary_task,
    train_meta_classifier,
    train_multiclass_classifier,
)
from classiflow.projects.dataset_registry import verify_manifest_hash
from classiflow.projects.project_fs import ProjectPaths
from classiflow.projects.project_models import ProjectConfig, DatasetRegistry, ThresholdsConfig
from classiflow.projects.reporting import write_technical_report, write_test_report
from classiflow.projects.promotion import normalize_metric_name
from classiflow.projects.final_train import (
    FinalTrainConfig,
    SelectedBinaryConfig,
    SelectedMetaConfig,
    extract_selected_configs_from_technical_run,
    save_selected_configs,
    train_final_meta_model,
    run_sanity_checks,
    validate_sanity_checks,
)

logger = logging.getLogger(__name__)


def _git_hash(cwd: Path) -> Optional[str]:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _config_hash(data: Dict) -> str:
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_id(phase: str, config_hash: str, train_hash: str, test_hash: Optional[str]) -> str:
    base = f"{phase}:{config_hash}:{train_hash}:{test_hash or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]


def _resolve_manifest(project_root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = project_root / manifest_path
    return path


def _lineage_payload(
    phase: str,
    run_id: str,
    config_hash: str,
    train_hash: str,
    test_hash: Optional[str],
    command: str,
    args: Dict[str, str],
    root: Path,
    outputs: Optional[List[Path]] = None,
) -> Dict:
    timestamp_local = datetime.now().isoformat()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "phase": phase,
        "run_id": run_id,
        "timestamp_local": timestamp_local,
        "timestamp_utc": timestamp_utc,
        "classiflow_version": __version__,
        "git_hash": _git_hash(root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_hash": config_hash,
        "dataset_hashes": {
            "train": train_hash,
            "test": test_hash,
        },
        "command": command,
        "args": args,
        "outputs": [],
    }
    if outputs:
        for item in outputs:
            if item.exists() and item.is_file():
                payload["outputs"].append(
                    {
                        "path": str(item),
                        "sha256": compute_file_hash(item),
                    }
                )
    return payload


def _load_registry(paths: ProjectPaths) -> DatasetRegistry:
    return DatasetRegistry.load(paths.datasets_yaml)


def _select_metric_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    metric = normalize_metric_name(metric)
    for col in df.columns:
        if normalize_metric_name(col) == metric:
            return col
    return None


def _summarize_metrics_from_df(
    df: pd.DataFrame, metrics: List[str]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    summary: Dict[str, float] = {}
    per_fold: Dict[str, List[float]] = {}
    for metric in metrics:
        col = _select_metric_column(df, metric)
        if not col:
            continue
        values = df[col].dropna().astype(float).tolist()
        if not values:
            continue
        per_fold[metric] = values
        summary[normalize_metric_name(metric)] = float(np.mean(values))

    if summary:
        return summary, per_fold

    exclude_cols = {"fold", "phase", "task", "model_name", "sampler", "level"}
    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in numeric_cols:
        values = df[col].dropna().astype(float).tolist()
        if not values:
            continue
        per_fold[col] = values
        summary[normalize_metric_name(col)] = float(np.mean(values))

    return summary, per_fold


def _append_metrics_from_df(
    df: pd.DataFrame,
    summary: Dict[str, float],
    per_fold: Dict[str, List[float]],
    metrics: List[str],
) -> None:
    for metric in metrics:
        if metric in summary:
            continue
        col = _select_metric_column(df, metric)
        if not col:
            continue
        values = df[col].dropna().astype(float).tolist()
        if not values:
            continue
        per_fold[metric] = values
        summary[normalize_metric_name(metric)] = float(np.mean(values))


def _technical_metrics_from_run(
    run_dir: Path, mode: str, metrics: List[str]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    if mode == "binary":
        metrics_path = run_dir / "metrics_outer_binary_eval.csv"
        df = pd.read_csv(metrics_path)
        df = df[df["phase"] == "val"]
        summary, per_fold = _summarize_metrics_from_df(df, metrics)
        _append_metrics_from_df(
            df,
            summary,
            per_fold,
            ["sensitivity", "specificity", "ppv", "npv", "recall", "precision", "mcc"],
        )
        pq_summary, pq_per_fold = _probability_quality_metrics_from_manifest(run_dir)
        summary.update(pq_summary)
        per_fold.update(pq_per_fold)
        return summary, per_fold
    if mode == "meta":
        metrics_path = run_dir / "metrics_outer_meta_eval.csv"
        df = pd.read_csv(metrics_path)
        df = df[df["phase"] == "val"]
        summary, per_fold = _summarize_metrics_from_df(df, metrics)
        _append_metrics_from_df(
            df,
            summary,
            per_fold,
            ["sensitivity", "specificity", "ppv", "npv", "recall", "precision", "mcc"],
        )
        cal_summary, cal_per_fold = _calibration_metrics_from_comparison(run_dir)
        if not cal_summary:
            cal_summary, cal_per_fold = _calibration_metrics_from_meta_df(df)
        summary.update(cal_summary)
        per_fold.update(cal_per_fold)
        pq_summary, pq_per_fold = _probability_quality_metrics_from_manifest(run_dir)
        summary.update(pq_summary)
        per_fold.update(pq_per_fold)
        return summary, per_fold
    if mode == "multiclass":
        metrics_path = run_dir / "metrics_outer_multiclass_eval.csv"
        df = pd.read_csv(metrics_path)
        df = df[df["phase"] == "val"]
        summary, per_fold = _summarize_metrics_from_df(df, metrics)
        _append_metrics_from_df(
            df,
            summary,
            per_fold,
            ["sensitivity", "specificity", "ppv", "npv", "recall", "precision", "mcc"],
        )
        pq_summary, pq_per_fold = _probability_quality_metrics_from_manifest(run_dir)
        summary.update(pq_summary)
        per_fold.update(pq_per_fold)
        return summary, per_fold

    metrics_path = run_dir / "metrics_outer_eval.csv"
    df = pd.read_csv(metrics_path)
    df = df[df["level"].isin(["pipeline", "L1"])]
    summary, per_fold = _summarize_metrics_from_df(df, metrics)
    _append_metrics_from_df(
        df,
        summary,
        per_fold,
        ["sensitivity", "specificity", "ppv", "npv", "recall", "precision", "mcc"],
    )
    pq_summary, pq_per_fold = _probability_quality_metrics_from_manifest(run_dir)
    summary.update(pq_summary)
    per_fold.update(pq_per_fold)
    return summary, per_fold


def _probability_quality_metrics_from_manifest(
    run_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    run_manifest_path = run_dir / "run.json"
    if not run_manifest_path.exists():
        return {}, {}
    try:
        run_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    prob_quality = ((run_payload.get("artifact_registry", {}) or {}).get("probability_quality", {}) or {})
    folds = prob_quality.get("folds")
    if not isinstance(folds, dict) or not folds:
        return {}, {}

    final_metric_values: Dict[str, List[float]] = {}
    variant_metric_values: Dict[str, List[float]] = {}
    for payload in folds.values():
        if not isinstance(payload, dict):
            continue
        final_variant = str(payload.get("final_variant") or "uncalibrated")
        final_metrics = payload.get(final_variant, {})
        if isinstance(final_metrics, dict):
            for key, value in final_metrics.items():
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                if value_float != value_float:
                    continue
                final_metric_values.setdefault(key, []).append(value_float)
        for variant in ("uncalibrated", "calibrated"):
            variant_metrics = payload.get(variant, {})
            if not isinstance(variant_metrics, dict):
                continue
            for key, value in variant_metrics.items():
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                if value_float != value_float:
                    continue
                variant_metric_values.setdefault(f"{key}_{variant}", []).append(value_float)

    summary = {key: float(np.mean(values)) for key, values in final_metric_values.items() if values}
    per_fold = {key: list(values) for key, values in final_metric_values.items() if values}
    for key, values in variant_metric_values.items():
        if not values:
            continue
        summary[key] = float(np.mean(values))
        per_fold[key] = list(values)
    if "brier_recommended_calibrated" in summary:
        summary["brier_calibrated"] = summary["brier_recommended_calibrated"]
        per_fold["brier_calibrated"] = list(per_fold.get("brier_recommended_calibrated", []))
    if "ece_top1_calibrated" in summary:
        summary["ece_calibrated"] = summary["ece_top1_calibrated"]
        per_fold["ece_calibrated"] = list(per_fold.get("ece_top1_calibrated", []))
    if "log_loss_calibrated" in summary:
        per_fold["log_loss_calibrated"] = list(per_fold.get("log_loss_calibrated", []))
    return summary, per_fold


def _calibration_metrics_from_meta_df(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    per_fold: Dict[str, List[float]] = {}
    summary: Dict[str, float] = {}
    candidates = [
        (["brier_recommended", "brier"], "brier_calibrated"),
        (["ece_top1", "ece"], "ece_calibrated"),
        (["log_loss"], "log_loss_calibrated"),
    ]
    for raw_keys, target in candidates:
        source_key = next((k for k in raw_keys if k in df.columns), None)
        if source_key is None:
            continue
        values = df[source_key].dropna().astype(float).tolist()
        if not values:
            continue
        per_fold[target] = values
        summary[target] = float(np.mean(values))
    return summary, per_fold


def _calibration_metrics_from_comparison(
    run_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    path = run_dir / "calibration_comparison.json"
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    selection = data.get("selection", {})
    method = selection.get("method_selected") or "sigmoid"
    per_fold: Dict[str, List[float]] = {
        "brier_calibrated": [],
        "ece_calibrated": [],
        "log_loss_calibrated": [],
    }
    for entry in data.get("folds", {}).values():
        metrics = entry.get(method, {})
        if not metrics or not metrics.get("enabled", False):
            continue
        for raw_keys, target in [
            (["brier_recommended", "brier"], "brier_calibrated"),
            (["ece_top1", "ece"], "ece_calibrated"),
            (["log_loss"], "log_loss_calibrated"),
        ]:
            value = None
            for key in raw_keys:
                if metrics.get(key) is not None:
                    value = metrics.get(key)
                    break
            if value is not None:
                per_fold[target].append(float(value))

    summary = {}
    for key, values in per_fold.items():
        if values:
            summary[key] = float(np.mean(values))
    return summary, per_fold


def _resolve_probability_quality_check_thresholds(config: ProjectConfig) -> Dict[str, Any]:
    policy = config.calibration.policy or {}
    thresholds: Dict[str, Any] = {}
    pq_cfg = policy.get("probability_quality_checks")
    if isinstance(pq_cfg, dict):
        nested = pq_cfg.get("thresholds")
        if isinstance(nested, dict):
            thresholds.update(dict(nested))
        else:
            # Backward-compatible shape where threshold keys lived directly under probability_quality_checks.
            thresholds.update(
                {
                    key: value
                    for key, value in pq_cfg.items()
                    if key not in {"enabled", "apply_to_modes"}
                }
            )
    policy_thresholds = policy.get("thresholds", {})
    if isinstance(policy_thresholds, dict):
        if isinstance(policy_thresholds.get("probability_quality_checks"), dict):
            thresholds.update(dict(policy_thresholds.get("probability_quality_checks", {})))
    return thresholds


def _probability_quality_checks_enabled(config: ProjectConfig) -> bool:
    policy = config.calibration.policy or {}
    pq_cfg = policy.get("probability_quality_checks")
    default_apply = ["binary", "multiclass", "hierarchical", "meta"]
    apply_modes = default_apply
    enabled = True
    if isinstance(pq_cfg, dict):
        enabled = bool(pq_cfg.get("enabled", True))
        configured_apply = pq_cfg.get("apply_to_modes")
        if isinstance(configured_apply, list) and configured_apply:
            apply_modes = [str(item) for item in configured_apply]
    elif isinstance(policy.get("apply_to_modes"), list):
        apply_modes = [str(item) for item in policy.get("apply_to_modes", default_apply)]
    return enabled and (config.task.mode in set(apply_modes))


def _configured_probability_quality_gate_metrics(thresholds: ThresholdsConfig) -> List[str]:
    metrics = [gate.metric for gate in thresholds.promotion_gates]
    calibration_cfg = thresholds.promotion.calibration
    if calibration_cfg.brier_max is not None:
        metrics.append("brier_calibrated")
    if calibration_cfg.ece_max is not None:
        metrics.append("ece_calibrated")
    return sorted(set(metrics))


def run_technical_validation(
    paths: ProjectPaths,
    config: ProjectConfig,
    run_id: Optional[str] = None,
    compare_smote: bool = False,
) -> Path:
    """Run technical validation using project configuration."""
    registry = _load_registry(paths)
    train_entry = registry.datasets.get("train")
    test_entry = registry.datasets.get("test")
    if not train_entry:
        raise ValueError("Training dataset not registered")
    if train_entry.dirty or (test_entry and test_entry.dirty):
        raise ValueError("Dataset registry marked dirty; re-register before running validation")

    train_manifest = _resolve_manifest(paths.root, config.data.train.manifest)
    test_manifest = None
    if config.data.test is not None:
        test_manifest = _resolve_manifest(paths.root, config.data.test.manifest)

    if not verify_manifest_hash(train_manifest, train_entry.sha256):
        raise ValueError("Training manifest hash mismatch; re-register dataset")
    if test_entry and test_manifest and not verify_manifest_hash(test_manifest, test_entry.sha256):
        raise ValueError("Test manifest hash mismatch; re-register dataset")

    config_hash = _config_hash(config.model_dump(mode="python"))
    run_id = run_id or _run_id(
        "technical_validation",
        config_hash,
        train_entry.sha256,
        test_entry.sha256 if test_entry else None,
    )
    run_dir = paths.runs_subdir("technical_validation", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.task.mode == "binary":
        train_config = TrainConfig(
            data_csv=train_manifest,
            label_col=config.key_columns.label,
            patient_col=config.key_columns.patient_id if config.task.patient_stratified else None,
            outdir=run_dir,
            outer_folds=config.validation.nested_cv.outer_folds,
            inner_splits=config.validation.nested_cv.inner_folds,
            inner_repeats=config.validation.nested_cv.repeats,
            random_state=config.validation.nested_cv.seed,
            smote_mode="both"
            if config.imbalance.smote.get("compare")
            else ("on" if config.imbalance.smote.get("enabled") else "off"),
            calibration_bins=config.calibration.bins,
            calibration_binning=config.calibration.binning,
            backend=config.backend,
            device=config.device,
            model_set=config.model_set,
            torch_num_workers=config.torch_num_workers,
            torch_dtype=config.torch_dtype,
            require_torch_device=config.require_torch_device,
            tracker=config.tracker,
            experiment_name=config.experiment_name or f"{config.project.id}/technical",
            run_name=f"technical-{run_id}",
        )
        train_binary_task(train_config)
    elif config.task.mode == "meta":
        train_config = MetaConfig(
            data_csv=train_manifest,
            label_col=config.key_columns.label,
            patient_col=config.key_columns.patient_id if config.task.patient_stratified else None,
            outdir=run_dir,
            outer_folds=config.validation.nested_cv.outer_folds,
            inner_splits=config.validation.nested_cv.inner_folds,
            inner_repeats=config.validation.nested_cv.repeats,
            random_state=config.validation.nested_cv.seed,
            smote_mode="both"
            if config.imbalance.smote.get("compare")
            else ("on" if config.imbalance.smote.get("enabled") else "off"),
            backend=config.backend,
            device=config.device,
            model_set=config.model_set,
            torch_num_workers=config.torch_num_workers,
            torch_dtype=config.torch_dtype,
            require_torch_device=config.require_torch_device,
            calibrate_meta=config.calibration.enabled != "false",
            calibration_enabled=config.calibration.enabled,
            calibration_method=config.calibration.method,
            calibration_cv=config.calibration.cv,
            calibration_bins=config.calibration.bins,
            calibration_binning=config.calibration.binning,
            calibration_isotonic_min_samples=config.calibration.isotonic_min_samples,
            calibration_policy_apply_to_modes=list(
                (config.calibration.policy or {}).get(
                    "apply_to_modes",
                    ["binary", "multiclass", "hierarchical", "meta"],
                )
            ),
            calibration_policy_force_keep=bool(
                (config.calibration.policy or {}).get("force_keep", False)
            ),
            calibration_policy_thresholds=dict(
                (config.calibration.policy or {}).get("thresholds", {})
            ),
            tracker=config.tracker,
            experiment_name=config.experiment_name or f"{config.project.id}/technical",
            run_name=f"technical-{run_id}",
        )
        train_meta_classifier(train_config)
    elif config.task.mode == "multiclass":
        train_config = MulticlassConfig(
            data_csv=train_manifest,
            label_col=config.key_columns.label,
            patient_col=config.key_columns.patient_id if config.task.patient_stratified else None,
            outdir=run_dir,
            outer_folds=config.validation.nested_cv.outer_folds,
            inner_splits=config.validation.nested_cv.inner_folds,
            inner_repeats=config.validation.nested_cv.repeats,
            random_state=config.validation.nested_cv.seed,
            smote_mode="both"
            if config.imbalance.smote.get("compare")
            else ("on" if config.imbalance.smote.get("enabled") else "off"),
            group_stratify=config.multiclass.group_stratify,
            estimator_mode=config.multiclass.estimator_mode,
            logreg_solver=config.multiclass.logreg.solver,
            logreg_multi_class=config.multiclass.logreg.multi_class,
            # penalty deprecated in sklearn 1.8; keep default and tune via l1_ratio/C
            logreg_max_iter=config.multiclass.logreg.max_iter,
            logreg_tol=config.multiclass.logreg.tol,
            logreg_C=config.multiclass.logreg.C,
            logreg_class_weight=config.multiclass.logreg.class_weight,
            logreg_n_jobs=config.multiclass.logreg.n_jobs,
            calibration_bins=config.calibration.bins,
            calibration_binning=config.calibration.binning,
            device=config.device,
            torch_num_workers=config.torch_num_workers,
            tracker=config.tracker,
            experiment_name=config.experiment_name or f"{config.project.id}/technical",
            run_name=f"technical-{run_id}",
        )
        train_multiclass_classifier(train_config)
    else:
        train_config = HierarchicalConfig(
            data_csv=train_manifest,
            patient_col=config.key_columns.patient_id if config.task.patient_stratified else None,
            label_l1=config.key_columns.label,
            label_l2=config.task.hierarchy_path,
            outdir=run_dir,
            outer_folds=config.validation.nested_cv.outer_folds,
            inner_splits=config.validation.nested_cv.inner_folds,
            random_state=config.validation.nested_cv.seed,
            use_smote=config.imbalance.smote.get("enabled", False),
            calibration_bins=config.calibration.bins,
            calibration_binning=config.calibration.binning,
            output_format="csv",
            tracker=config.tracker,
            experiment_name=config.experiment_name or f"{config.project.id}/technical",
            run_name=f"technical-{run_id}",
        )
        from classiflow.training.hierarchical_cv import train_hierarchical

        train_hierarchical(train_config)

    if config.imbalance.smote.get("compare") or compare_smote:
        comparison = SMOTEComparison.from_directory(run_dir)
        report = comparison.generate_report()
        smote_dir = run_dir / "smote_comparison"
        comparison.save_report(report, smote_dir)
        comparison.create_all_plots(outdir=smote_dir)

    config_path = run_dir / "config.resolved.yaml"
    config.save(config_path)

    summary, per_fold = _technical_metrics_from_run(
        run_dir, config.task.mode, config.metrics.primary
    )
    notes: List[str] = []
    calibration_decision: Optional[Dict[str, Any]] = None
    if config.task.mode == "meta":
        meta_metrics_path = run_dir / "metrics_outer_meta_eval.csv"
        if meta_metrics_path.exists():
            meta_df = pd.read_csv(meta_metrics_path)
            methods = sorted(set(meta_df["calibration_method"].dropna().astype(str).tolist()))
            if methods:
                notes.append(f"Calibration method(s): {', '.join(methods)}")
            bins = sorted(set(meta_df["calibration_bins"].dropna().astype(int).tolist()))
            if bins:
                notes.append(f"Calibration bins: {', '.join(map(str, bins))}")
        comparison_path = run_dir / "calibration_comparison.json"
        if comparison_path.exists():
            try:
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                selection = comparison.get("selection", {})
                selected = selection.get("method_selected")
                reason = selection.get("reason")
                if selected:
                    notes.append(f"Calibration selection: {selected}")
                if reason:
                    notes.append(f"Calibration selection rationale: {reason}")
            except Exception:
                pass
        run_manifest_path = run_dir / "run.json"
        if run_manifest_path.exists():
            try:
                run_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                prob_quality = (
                    (run_payload.get("artifact_registry", {}) or {}).get("probability_quality", {})
                )
                folds = prob_quality.get("folds", {}) if isinstance(prob_quality, dict) else {}
                if isinstance(folds, dict) and folds:
                    first_fold = next(iter(folds.values()))
                    if isinstance(first_fold, dict):
                        decision = dict(first_fold.get("calibration_decision", {}) or {})
                        decision["final_variant"] = first_fold.get("final_variant")
                        decision["pred_alignment_mismatch_rate"] = (
                            (first_fold.get("uncalibrated", {}) or {}).get(
                                "pred_alignment_mismatch_rate"
                            )
                            if first_fold.get("final_variant") == "uncalibrated"
                            else (first_fold.get("calibrated", {}) or {}).get(
                                "pred_alignment_mismatch_rate"
                            )
                        )
                        calibration_decision = decision
            except Exception:
                calibration_decision = None
    report_dir = run_dir / "reports"
    thresholds_cfg = ThresholdsConfig.load(paths.thresholds_yaml)
    pq_checks_enabled = _probability_quality_checks_enabled(config)
    write_technical_report(
        report_dir,
        summary,
        per_fold,
        notes=notes,
        calibration_decision=calibration_decision,
        run_dir=run_dir if pq_checks_enabled else None,
        task_mode=config.task.mode if pq_checks_enabled else None,
        probability_quality_check_thresholds=_resolve_probability_quality_check_thresholds(config),
        promotion_gate_metrics=_configured_probability_quality_gate_metrics(thresholds_cfg),
    )

    lineage = _lineage_payload(
        phase="TECHNICAL_VALIDATION",
        run_id=run_id,
        config_hash=config_hash,
        train_hash=train_entry.sha256,
        test_hash=test_entry.sha256 if test_entry else None,
        command="classiflow project run-technical",
        args={"run_id": run_id},
        root=paths.root,
        outputs=[config_path],
    )
    with open(run_dir / "lineage.json", "w", encoding="utf-8") as handle:
        json.dump(lineage, handle, indent=2)

    metrics_path = run_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_fold": per_fold}, handle, indent=2)

    return run_dir


def run_feasibility(
    paths: ProjectPaths,
    config: ProjectConfig,
    run_id: Optional[str] = None,
    classes: Optional[List[str]] = None,
    alpha: float = 0.05,
    min_n: int = 3,
    dunn_adjust: str = "holm",
    top_n: int = 30,
    write_legacy_csv: bool = True,
    write_legacy_xlsx: bool = True,
    run_viz: bool = True,
    fc_thresh: float = 1.0,
    fc_center: str = "median",
    label_topk: int = 12,
    heatmap_topn: int = 30,
    fig_dpi: int = 160,
) -> Path:
    """Run feasibility stats + visualizations on the training dataset."""
    registry = _load_registry(paths)
    train_entry = registry.datasets.get("train")
    if not train_entry:
        raise ValueError("Training dataset not registered")
    if train_entry.dirty:
        raise ValueError("Dataset registry marked dirty; re-register before running feasibility")

    train_manifest = _resolve_manifest(paths.root, config.data.train.manifest)
    if not verify_manifest_hash(train_manifest, train_entry.sha256):
        raise ValueError("Training manifest hash mismatch; re-register dataset")

    feasibility_options = {
        "classes": classes,
        "alpha": alpha,
        "min_n": min_n,
        "dunn_adjust": dunn_adjust,
        "top_n": top_n,
        "write_legacy_csv": write_legacy_csv,
        "write_legacy_xlsx": write_legacy_xlsx,
        "run_viz": run_viz,
        "fc_thresh": fc_thresh,
        "fc_center": fc_center,
        "label_topk": label_topk,
        "heatmap_topn": heatmap_topn,
        "fig_dpi": fig_dpi,
    }
    config_hash = _config_hash(
        {
            "project": config.model_dump(mode="python"),
            "feasibility": feasibility_options,
        }
    )
    run_id = run_id or _run_id("feasibility", config_hash, train_entry.sha256, None)
    run_dir = paths.runs_subdir("feasibility", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    from classiflow.stats import run_stats, run_visualizations

    stats_results = run_stats(
        data_csv=train_manifest,
        label_col=config.key_columns.label,
        outdir=run_dir,
        classes=classes,
        alpha=alpha,
        min_n=min_n,
        dunn_adjust=dunn_adjust,
        top_n_features=top_n,
        write_legacy_csv=write_legacy_csv,
        write_legacy_xlsx=write_legacy_xlsx,
    )

    viz_results = None
    if run_viz:
        viz_results = run_visualizations(
            data_csv=train_manifest,
            label_col=config.key_columns.label,
            outdir=run_dir,
            stats_dir=stats_results["stats_dir"],
            classes=classes,
            alpha=alpha,
            fc_thresh=fc_thresh,
            fc_center=fc_center,
            label_topk=label_topk,
            heatmap_topn=heatmap_topn,
            fig_dpi=fig_dpi,
        )

    config_path = run_dir / "feasibility_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_csv": str(train_manifest),
                "label_col": config.key_columns.label,
                "run_viz": run_viz,
                "stats_options": feasibility_options,
                "stats_outputs": {
                    "stats_dir": str(stats_results.get("stats_dir")),
                    "publication_xlsx": str(stats_results.get("publication_xlsx")),
                    "legacy_xlsx": str(stats_results.get("legacy_xlsx"))
                    if stats_results.get("legacy_xlsx")
                    else None,
                },
                "viz_outputs": {
                    "viz_dir": str(viz_results.get("viz_dir")) if viz_results else None,
                },
            },
            handle,
            indent=2,
        )

    metrics_path = run_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": {
                    "n_samples": stats_results["n_samples"],
                    "n_features": stats_results["n_features"],
                    "n_classes": stats_results["n_classes"],
                },
                "per_fold": {},
            },
            handle,
            indent=2,
        )

    outputs = [config_path, metrics_path, stats_results["publication_xlsx"]]
    if stats_results.get("legacy_xlsx"):
        outputs.append(stats_results["legacy_xlsx"])

    lineage = _lineage_payload(
        phase="FEASIBILITY",
        run_id=run_id,
        config_hash=config_hash,
        train_hash=train_entry.sha256,
        test_hash=None,
        command="classiflow project run-feasibility",
        args={"run_id": run_id},
        root=paths.root,
        outputs=outputs,
    )
    with open(run_dir / "lineage.json", "w", encoding="utf-8") as handle:
        json.dump(lineage, handle, indent=2)

    return run_dir


def _select_best_binary_config(
    metrics_path: Path, selection_metric: str, direction: str
) -> Dict[str, str]:
    df = pd.read_csv(metrics_path)
    metric_col = _select_metric_column(df, selection_metric)
    if metric_col is None:
        metric_col = "mean_test_score" if "mean_test_score" in df.columns else "mean_test_f1"
    df = df.dropna(subset=[metric_col])
    sort_asc = direction == "min"
    df = df.sort_values(metric_col, ascending=sort_asc)
    best = df.iloc[0]
    params = {
        k: best[k]
        for k in df.columns
        if k
        not in {
            "fold",
            "sampler",
            "task",
            "model_name",
            "rank_test_score",
            "rank_test_f1",
            "mean_test_f1",
            "std_test_f1",
            "mean_test_score",
            "std_test_score",
            metric_col,
        }
    }
    return {
        "model_name": best["model_name"],
        "sampler": best.get("sampler", "none"),
        "params": params,
    }


def _select_best_multiclass_config(
    metrics_path: Path, selection_metric: str, direction: str
) -> Dict[str, str]:
    df = pd.read_csv(metrics_path)
    metric_col = _select_metric_column(df, selection_metric)
    if metric_col is None:
        metric_col = (
            "mean_test_f1_macro" if "mean_test_f1_macro" in df.columns else "mean_test_score"
        )
    df = df.dropna(subset=[metric_col])
    sort_asc = direction == "min"
    df = df.sort_values(metric_col, ascending=sort_asc)
    best = df.iloc[0]
    params = {
        k: best[k]
        for k in df.columns
        if k
        not in {
            "fold",
            "sampler",
            "task",
            "model_name",
            "rank_test_f1_macro",
            "mean_test_f1_macro",
            "std_test_f1_macro",
            "mean_test_score",
            "std_test_score",
            metric_col,
        }
    }
    return {
        "model_name": best["model_name"],
        "sampler": best.get("sampler", "none"),
        "params": params,
    }


def _filter_model_params(estimator, params: Dict[str, object]) -> Dict[str, object]:
    valid = set(estimator.get_params().keys())
    cleaned: Dict[str, object] = {}
    defaults = estimator.get_params()
    int_param_hints = {
        "hidden_dim",
        "n_layers",
        "epochs",
        "batch_size",
        "max_iter",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    }
    for key, value in params.items():
        if key not in valid:
            continue
        if value != value:
            continue
        default = defaults.get(key)
        if isinstance(value, float) and value.is_integer():
            if isinstance(default, int) or key in int_param_hints:
                value = int(value)
        if isinstance(default, bool) and isinstance(value, (int, float)):
            value = bool(value)
        cleaned[key] = value
    return cleaned


def _train_final_binary(
    config: ProjectConfig,
    train_manifest: Path,
    outdir: Path,
    best_cfg: Dict[str, str],
) -> str:
    from classiflow.io import load_data

    X, y_raw = load_data(train_manifest, config.key_columns.label)
    pos_label = y_raw.value_counts().idxmin()
    y = (y_raw == pos_label).astype(int)

    estimator = get_estimators(config.validation.nested_cv.seed, 10000)[best_cfg["model_name"]]
    cleaned = _filter_model_params(estimator, best_cfg["params"])
    params = {f"clf__{k}": v for k, v in cleaned.items() if k}
    sampler = AdaptiveSMOTE(k_max=5, random_state=config.validation.nested_cv.seed)
    if best_cfg.get("sampler") == "none":
        sampler = "passthrough"

    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline(
        [
            ("sampler", sampler),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
    if params:
        pipe.set_params(**params)
    pipe.fit(X, y)

    fold_dir = (
        outdir / "fold1" / f"binary_{'smote' if best_cfg.get('sampler') == 'smote' else 'none'}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipes": {"binary_task__" + best_cfg["model_name"]: pipe},
            "best_models": {"binary_task": best_cfg["model_name"]},
        },
        fold_dir / "binary_pipes.joblib",
    )
    return str(pos_label)


def _train_final_meta(
    config: ProjectConfig,
    train_manifest: Path,
    outdir: Path,
    best_cfg: Dict[str, str],
    technical_run: Optional[Path] = None,
) -> None:
    """
    Train final meta-classifier for deployment.

    IMPORTANT: This function now reuses binary pipelines from technical validation
    rather than retraining with potentially different hyperparameters. This ensures
    the final model has the same architecture as what was validated during CV.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration
    train_manifest : Path
        Path to training data manifest
    outdir : Path
        Output directory for final model artifacts
    best_cfg : Dict[str, str]
        Best configuration from inner CV (used for sampler selection)
    technical_run : Path, optional
        Path to technical validation run directory. If provided, binary pipelines
        are copied from this run. Otherwise, pipelines are retrained (legacy behavior).
    """
    from classiflow.io import load_data
    from sklearn.calibration import CalibratedClassifierCV

    X_full, y_full = load_data(train_manifest, config.key_columns.label)
    classes = sorted(y_full.unique().tolist())
    task_builder = TaskBuilder(classes).build_all_auto_tasks()
    tasks = task_builder.get_tasks()

    variant = "smote" if best_cfg.get("sampler") == "smote" else "none"

    # Try to load binary pipelines from technical validation
    # This ensures we use the EXACT same trained models that were validated
    best_pipes = None
    best_models = None
    reused_from_technical = False

    if technical_run is not None:
        tech_fold_dir = technical_run / "fold1" / f"binary_{variant}"
        tech_pipes_path = tech_fold_dir / "binary_pipes.joblib"

        if tech_pipes_path.exists():
            logger.info(f"Reusing binary pipelines from technical validation: {tech_pipes_path}")
            try:
                bundle = joblib.load(tech_pipes_path)
                if isinstance(bundle, dict):
                    best_pipes = bundle.get("pipes", {})
                    best_models = bundle.get("best_models", {})
                    reused_from_technical = True
                    logger.info(
                        f"  Loaded {len(best_pipes)} pipelines, {len(best_models)} best models"
                    )

                    # Validate that pipelines produce meaningful predictions
                    _validate_binary_pipelines(best_pipes, best_models, X_full, y_full, tasks)
                else:
                    logger.warning("  Unexpected format in binary_pipes.joblib, will retrain")
            except Exception as exc:
                logger.warning(f"  Failed to load binary pipelines: {exc}, will retrain")

    # Fallback: retrain binary pipelines if not loaded from technical validation
    if best_pipes is None or best_models is None:
        logger.warning("Retraining binary pipelines (not reusing from technical validation)")
        best_pipes, best_models = _retrain_binary_pipelines_per_task(
            config, X_full, y_full, tasks, technical_run, variant
        )

    from classiflow.training.meta import _build_meta_features
    from sklearn.linear_model import LogisticRegression

    X_meta = _build_meta_features(X_full, y_full, best_pipes, best_models, tasks, config)

    # Validate meta-features have reasonable variance
    _validate_meta_features(X_meta)

    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=10000,
        random_state=config.validation.nested_cv.seed,
    )
    meta_model.fit(X_meta.values, y_full.values)
    calibration_metadata = {
        "enabled": False,
        "method_requested": config.calibration.method,
        "method_used": None,
        "cv": None,
        "bins": config.calibration.bins,
        "warnings": [],
        "reused_binary_pipelines_from_technical": reused_from_technical,
    }
    if config.calibration.enabled == "false":
        calibration_metadata["warnings"].append("Calibration disabled in configuration.")
    else:
        method = config.calibration.method
        y_series = pd.Series(y_full)
        if method == "isotonic":
            min_samples = config.calibration.isotonic_min_samples
            if len(X_meta) < min_samples or y_series.value_counts().min() < 2:
                calibration_metadata["warnings"].append(
                    "Isotonic calibration not supported (min samples or class counts too low); falling back to sigmoid."
                )
                method = "sigmoid"
        cv = max(2, min(config.calibration.cv, max(2, len(X_meta) - 1)))
        try:
            calibrator = CalibratedClassifierCV(
                estimator=meta_model,
                method=method,
                cv=cv,
            )
            calibrator.fit(X_meta.values, y_full.values)
            meta_model = calibrator
            calibration_metadata.update(
                {
                    "enabled": True,
                    "method_used": method,
                    "cv": cv,
                }
            )
        except Exception as exc:
            calibration_metadata["warnings"].append(f"Calibration failed: {exc}")
            calibration_metadata.update(
                {
                    "enabled": False,
                    "method_used": method,
                    "cv": cv,
                }
            )

    fold_dir = outdir / "fold1" / f"binary_{variant}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipes": best_pipes, "best_models": best_models}, fold_dir / "binary_pipes.joblib")
    joblib.dump(meta_model, fold_dir / "meta_model.joblib")
    pd.Series(list(X_meta.columns)).to_csv(
        fold_dir / "meta_features.csv", index=False, header=False
    )
    pd.Series(list(meta_model.classes_)).to_csv(
        fold_dir / "meta_classes.csv", index=False, header=False
    )
    with open(fold_dir / "calibration_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(calibration_metadata, handle, indent=2)


def _validate_binary_pipelines(
    best_pipes: Dict[str, Any],
    best_models: Dict[str, str],
    X_full: pd.DataFrame,
    y_full: pd.Series,
    tasks: Dict[str, Callable],
) -> None:
    """
    Validate that binary pipelines produce discriminative predictions.

    Raises ValueError if pipelines appear to be producing near-random outputs.
    """
    MIN_STD_THRESHOLD = (
        0.05  # Minimum standard deviation for predictions to be considered meaningful
    )
    MAX_MEAN_DEVIATION = 0.15  # Maximum deviation from 0.5 for mean to indicate random predictions

    for task_name, model_name in best_models.items():
        key = f"{task_name}__{model_name}"
        if key not in best_pipes:
            continue

        pipe = best_pipes[key]
        y_bin = tasks[task_name](y_full).dropna()
        if y_bin.empty:
            continue

        X_subset = X_full.loc[y_bin.index]

        # Get predictions
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_subset)
            scores = proba[:, 1] if proba.ndim > 1 else proba
        else:
            scores = pipe.decision_function(X_subset)

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Check for near-random predictions
        is_near_random = (
            std_score < MIN_STD_THRESHOLD and abs(mean_score - 0.5) < MAX_MEAN_DEVIATION
        )

        if is_near_random:
            raise ValueError(
                f"Binary pipeline '{key}' appears to produce near-random predictions "
                f"(mean={mean_score:.4f}, std={std_score:.4f}). "
                f"This indicates the model was not trained properly or has different architecture. "
                f"Check that the technical validation pipelines are compatible."
            )

        logger.debug(f"  Validated {key}: mean={mean_score:.4f}, std={std_score:.4f}")


def _validate_meta_features(X_meta: pd.DataFrame) -> None:
    """
    Validate that meta-features have reasonable variance.

    Raises ValueError if meta-features appear to be degenerate.
    """
    MIN_STD_THRESHOLD = 0.01

    for col in X_meta.columns:
        std_val = X_meta[col].std()
        if std_val < MIN_STD_THRESHOLD:
            mean_val = X_meta[col].mean()
            if abs(mean_val - 0.5) < 0.1:
                raise ValueError(
                    f"Meta-feature '{col}' has very low variance (std={std_val:.6f}, mean={mean_val:.4f}). "
                    f"This indicates the underlying binary classifier is not producing meaningful predictions."
                )


def _retrain_binary_pipelines_per_task(
    config: ProjectConfig,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    tasks: Dict[str, Callable],
    technical_run: Optional[Path],
    variant: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Retrain binary pipelines using PER-TASK best configurations from inner CV.

    This fixes the bug where the global best configuration was used for all tasks,
    which could result in suboptimal or failed models.
    """
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline as ImbPipeline

    best_pipes = {}
    best_models = {}

    # Load per-task best configurations from inner CV
    task_configs = {}
    if technical_run is not None:
        metrics_path = technical_run / "metrics_inner_cv.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            df = df[df["sampler"] == variant]

            for task_name in df["task"].unique():
                task_df = df[df["task"] == task_name]
                task_df = task_df.dropna(subset=["mean_test_score"])
                if task_df.empty:
                    continue

                best_row = task_df.sort_values("mean_test_score", ascending=False).iloc[0]
                task_configs[task_name] = {
                    "model_name": best_row["model_name"],
                    "params": {
                        k: best_row[k]
                        for k in best_row.index
                        if k
                        not in {
                            "fold",
                            "sampler",
                            "task",
                            "model_name",
                            "rank_test_score",
                            "mean_test_score",
                            "std_test_score",
                        }
                        and pd.notna(best_row[k])
                    },
                }
            logger.info(f"Loaded per-task configurations for {len(task_configs)} tasks")

    # Get model registry
    model_spec = get_model_set(
        command="train-meta",
        backend=get_backend(config.backend),
        model_set=config.model_set,
        random_state=config.validation.nested_cv.seed,
        max_iter=config.multiclass.logreg.max_iter,
        device=config.device,
        torch_dtype=config.torch_dtype,
        torch_num_workers=config.torch_num_workers,
        meta_C_grid=None,
    )
    estimators = model_spec["base_estimators"]

    sampler = (
        AdaptiveSMOTE(k_max=5, random_state=config.validation.nested_cv.seed)
        if variant == "smote"
        else "passthrough"
    )

    for task_name, task_func in tasks.items():
        y_bin = task_func(y_full).dropna()
        if y_bin.nunique() < 2:
            continue
        X_bin = X_full.loc[y_bin.index]

        # Get per-task config or fall back to first available model
        if task_name in task_configs:
            task_cfg = task_configs[task_name]
            model_name = task_cfg["model_name"]
            params = task_cfg["params"]
        else:
            # Fallback: use first estimator
            model_name = list(estimators.keys())[0]
            params = {}
            logger.warning(f"No per-task config for {task_name}, using default {model_name}")

        if model_name not in estimators:
            logger.warning(f"Model {model_name} not in estimators, skipping {task_name}")
            continue

        estimator = estimators[model_name]
        cleaned = _filter_model_params(estimator, params)
        clf_params = {f"clf__{k}": v for k, v in cleaned.items() if k}

        pipe = ImbPipeline(
            [
                ("sampler", sampler),
                ("scaler", StandardScaler()),
                ("clf", estimator),
            ]
        )
        if clf_params:
            pipe.set_params(**clf_params)

        try:
            pipe.fit(X_bin, y_bin)
            best_pipes[f"{task_name}__{model_name}"] = pipe
            best_models[task_name] = model_name
            logger.info(f"  Trained {task_name} with {model_name}")
        except Exception as exc:
            logger.warning(f"  Failed to train {task_name} with {model_name}: {exc}")

    return best_pipes, best_models


def _train_final_multiclass(
    config: ProjectConfig,
    train_manifest: Path,
    outdir: Path,
    best_cfg: Dict[str, str],
) -> None:
    from classiflow.io import load_data
    from classiflow.models import get_estimators, resolve_device

    X_full, y_full = load_data(train_manifest, config.key_columns.label)
    classes = sorted(y_full.unique().tolist())

    logreg_params = {
        "solver": config.multiclass.logreg.solver,
        # penalty deprecated in sklearn 1.8; keep default and tune via l1_ratio/C
        "max_iter": config.multiclass.logreg.max_iter,
        "tol": config.multiclass.logreg.tol,
        "C": config.multiclass.logreg.C,
        "class_weight": config.multiclass.logreg.class_weight,
        "n_jobs": config.multiclass.logreg.n_jobs,
    }
    if config.multiclass.logreg.multi_class and config.multiclass.logreg.multi_class != "auto":
        logreg_params["multi_class"] = config.multiclass.logreg.multi_class

    resolved_device = resolve_device(config.device)
    estimators = get_estimators(
        config.validation.nested_cv.seed,
        10000,
        logreg_params=logreg_params,
        resolved_device=resolved_device,
        torch_num_workers=config.torch_num_workers,
    )
    estimator_mode = config.multiclass.estimator_mode
    if estimator_mode == "torch_only":
        estimators = {k: v for k, v in estimators.items() if k.startswith("torch_")}
    elif estimator_mode == "cpu_only":
        estimators = {k: v for k, v in estimators.items() if not k.startswith("torch_")}
    elif estimator_mode != "all":
        raise ValueError(f"Unsupported multiclass estimator_mode: {estimator_mode}")
    if not estimators:
        raise ValueError(
            f"No estimators available for estimator_mode={estimator_mode} "
            f"with resolved_device={resolved_device}."
        )

    estimator = estimators[best_cfg["model_name"]]
    cleaned = _filter_model_params(estimator, best_cfg["params"])
    params = {f"clf__{k}": v for k, v in cleaned.items() if k}
    sampler = AdaptiveSMOTE(k_max=5, random_state=config.validation.nested_cv.seed)
    if best_cfg.get("sampler") == "none":
        sampler = "passthrough"

    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline(
        [
            ("sampler", sampler),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
    if params:
        pipe.set_params(**params)
    pipe.fit(X_full, y_full)

    variant = "smote" if best_cfg.get("sampler") == "smote" else "none"
    fold_dir = outdir / "fold1" / f"multiclass_{variant}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, fold_dir / "multiclass_model.joblib")
    (fold_dir / "multiclass_model_name.txt").write_text(best_cfg["model_name"])
    pd.Series(classes).to_csv(fold_dir / "classes.csv", index=False, header=False)
    pd.Series(list(X_full.columns)).to_csv(fold_dir / "feature_list.csv", index=False, header=False)


def _train_final_hierarchical(
    config: ProjectConfig,
    train_manifest: Path,
    outdir: Path,
    metrics_path: Path,
) -> None:
    import json as jsonlib
    import numpy as np
    from classiflow.data import load_table
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from classiflow.models.smote import apply_smote

    df = load_table(train_manifest)
    label_l1 = config.key_columns.label
    label_l2 = config.task.hierarchy_path
    if label_l2 is None:
        raise ValueError("hierarchy_path required for hierarchical final model")

    X_all = df.select_dtypes(include=[np.number]).values
    y_l1_all = df[label_l1].astype(str).values
    y_l2_all = df[label_l2].astype(str).values

    le_l1 = LabelEncoder().fit(y_l1_all)
    y_l1_enc = le_l1.transform(y_l1_all)

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=config.validation.nested_cv.seed
    )
    train_idx, val_idx = next(split.split(X_all, y_l1_enc))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    X_val = scaler.transform(X_all[val_idx])

    l1_classes = le_l1.classes_.tolist()

    inner_df = pd.read_csv(metrics_path)
    l1_rows = inner_df[inner_df["level"] == "L1"]
    if l1_rows.empty:
        raise ValueError("No L1 hyperparameter results found")
    l1_rows = l1_rows.sort_values("mean_f1_macro", ascending=False)
    best_cfg_l1 = l1_rows.iloc[0]

    best_cfg = {
        "hidden_dims": jsonlib.loads(best_cfg_l1["hidden_dims"])
        if isinstance(best_cfg_l1["hidden_dims"], str)
        else best_cfg_l1["hidden_dims"],
        "lr": float(best_cfg_l1["lr"]),
        "epochs": int(best_cfg_l1["epochs"]),
        "dropout": float(best_cfg_l1["dropout"]),
    }

    X_train_l1 = X_train
    y_train_l1 = y_l1_enc[train_idx]
    if config.imbalance.smote.get("enabled"):
        X_train_l1, y_train_l1 = apply_smote(
            X_train_l1, y_train_l1, 5, config.validation.nested_cv.seed
        )

    from classiflow.models.torch_mlp import TorchMLPWrapper

    model_l1 = TorchMLPWrapper(
        input_dim=X_train.shape[1],
        num_classes=len(l1_classes),
        hidden_dims=best_cfg["hidden_dims"],
        lr=best_cfg["lr"],
        epochs=best_cfg["epochs"],
        dropout=best_cfg["dropout"],
        batch_size=256,
        early_stopping_patience=10,
        device="auto",
        random_state=config.validation.nested_cv.seed,
        verbose=1,
    )
    model_l1.fit(X_train_l1, y_train_l1, X_val, y_l1_enc[val_idx])

    fold_dir = outdir / "fold1"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_l1.save(fold_dir / "model_level1_fold1.pt")
    with open(fold_dir / "model_config_l1_fold1.json", "w", encoding="utf-8") as handle:
        jsonlib.dump(model_l1.get_config(), handle, indent=2)
    joblib.dump(scaler, fold_dir / "scaler.joblib")
    joblib.dump(le_l1, fold_dir / "label_encoder_l1.joblib")

    # L2 branch training
    if label_l2:
        for l1_value in l1_classes:
            mask = y_l1_all == l1_value
            if mask.sum() < 5:
                continue
            l2_values = y_l2_all[mask]
            le_l2 = LabelEncoder().fit(l2_values)

            l2_rows = inner_df[inner_df["level"] == f"L2_{l1_value}"]
            if l2_rows.empty:
                continue
            l2_rows = l2_rows.sort_values("mean_f1_macro", ascending=False)
            best_cfg_l2 = l2_rows.iloc[0]
            cfg_l2 = {
                "hidden_dims": jsonlib.loads(best_cfg_l2["hidden_dims"])
                if isinstance(best_cfg_l2["hidden_dims"], str)
                else best_cfg_l2["hidden_dims"],
                "lr": float(best_cfg_l2["lr"]),
                "epochs": int(best_cfg_l2["epochs"]),
                "dropout": float(best_cfg_l2["dropout"]),
            }

            X_branch = scaler.transform(X_all[mask])
            y_branch = le_l2.transform(l2_values)

            split = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=config.validation.nested_cv.seed
            )
            tr_idx, va_idx = next(split.split(X_branch, y_branch))
            X_tr_b, X_va_b = X_branch[tr_idx], X_branch[va_idx]
            y_tr_b, y_va_b = y_branch[tr_idx], y_branch[va_idx]

            model_l2 = TorchMLPWrapper(
                input_dim=X_branch.shape[1],
                num_classes=len(le_l2.classes_),
                hidden_dims=cfg_l2["hidden_dims"],
                lr=cfg_l2["lr"],
                epochs=cfg_l2["epochs"],
                dropout=cfg_l2["dropout"],
                batch_size=256,
                early_stopping_patience=10,
                device="auto",
                random_state=config.validation.nested_cv.seed,
                verbose=1,
            )
            model_l2.fit(X_tr_b, y_tr_b, X_va_b, y_va_b)
            safe_l1 = l1_value.replace(" ", "_")
            model_l2.save(fold_dir / f"model_level2_{safe_l1}_fold1.pt")
            with open(
                fold_dir / f"model_config_l2_{safe_l1}_fold1.json", "w", encoding="utf-8"
            ) as handle:
                jsonlib.dump(model_l2.get_config(), handle, indent=2)
            joblib.dump(le_l2, fold_dir / f"label_encoder_l2_{safe_l1}.joblib")

    config_path = outdir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        training_config = HierarchicalConfig(
            data_csv=train_manifest,
            patient_col=config.key_columns.patient_id if config.task.patient_stratified else None,
            label_l1=label_l1,
            label_l2=label_l2,
            outdir=outdir,
            outer_folds=1,
            inner_splits=config.validation.nested_cv.inner_folds,
            random_state=config.validation.nested_cv.seed,
        )
        jsonlib.dump(training_config.to_dict(), handle, indent=2)


def build_final_model(
    paths: ProjectPaths,
    config: ProjectConfig,
    technical_run: Path,
    run_id: Optional[str] = None,
    sampler: Optional[str] = None,
) -> Path:
    """
    Build final model by training from scratch on 100% of training data.

    This function implements the revised final model workflow that:
    1. Extracts per-task best configs from technical validation
    2. Trains ALL models from scratch on full training data (no fold reuse)
    3. Runs sanity checks to detect degenerate predictions
    4. Creates a complete model bundle for inference

    Parameters
    ----------
    paths : ProjectPaths
        Project path configuration
    config : ProjectConfig
        Project configuration
    technical_run : Path
        Path to technical validation run directory
    run_id : Optional[str]
        Override run ID (auto-generated if None)
    sampler : Optional[str]
        Sampler to use for final training ("none", "smote", or None to auto-select)

    Returns
    -------
    run_dir : Path
        Path to final model run directory containing model_bundle.zip
    """
    logger.info("=" * 70)
    logger.info("BUILD FINAL MODEL (train from scratch)")
    logger.info("=" * 70)

    registry = _load_registry(paths)
    train_entry = registry.datasets.get("train")
    test_entry = registry.datasets.get("test")
    if not train_entry:
        raise ValueError("Training dataset not registered")

    train_manifest = _resolve_manifest(paths.root, config.data.train.manifest)

    # Verify dataset hash
    if not verify_manifest_hash(train_manifest, train_entry.sha256):
        raise ValueError(
            "Training manifest hash mismatch. The dataset has changed since registration. "
            "Re-register the dataset with 'classiflow project register-dataset --type train'"
        )

    # Apply calibration selection from technical validation
    effective_config = _apply_calibration_selection(config, technical_run)

    # Determine sampler variant
    if sampler is None:
        # Auto-select based on config or technical validation results
        if config.imbalance.smote.get("enabled"):
            sampler = "smote"
        else:
            sampler = "none"
            # Check if SMOTE was better in technical validation
            smote_comparison_path = technical_run / "smote_comparison" / "summary.json"
            if smote_comparison_path.exists():
                try:
                    smote_data = json.loads(smote_comparison_path.read_text())
                    if smote_data.get("recommendation") == "smote":
                        sampler = "smote"
                        logger.info("Auto-selected SMOTE based on technical validation comparison")
                except Exception:
                    pass

    logger.info(f"Sampler: {sampler}")
    logger.info(f"Technical run: {technical_run}")

    config_hash = _config_hash(
        {
            **effective_config.model_dump(mode="python"),
            "sampler": sampler,
        }
    )
    run_id = run_id or _run_id(
        "final_model",
        config_hash,
        train_entry.sha256,
        test_entry.sha256 if test_entry else None,
    )
    run_dir = paths.runs_subdir("final_model", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {run_dir}")

    # Extract per-task best configurations from technical validation
    logger.info("\n[1/4] Extracting validated configurations...")
    try:
        binary_configs, meta_config = extract_selected_configs_from_technical_run(
            technical_run=technical_run,
            variant=sampler,
            selection_metric=effective_config.models.selection_metric,
            direction=effective_config.models.selection_direction,
        )
        logger.info(f"  Extracted {len(binary_configs)} per-task configurations")
    except Exception as e:
        logger.warning(f"Could not extract configs from technical run: {e}")
        binary_configs = {}
        meta_config = None

    # Save selected configs as source of truth
    save_selected_configs(run_dir, binary_configs, meta_config)

    task_definitions: Dict[str, Any] = {}
    best_models: Dict[str, str] = {}

    # Mode-specific training
    if effective_config.task.mode == "meta":
        logger.info("\n[2/4] Training final meta-classifier (from scratch)...")

        # Create FinalTrainConfig
        final_config = FinalTrainConfig(
            train_manifest=train_manifest,
            label_col=effective_config.key_columns.label,
            mode="meta",
            classes=None,  # Use all classes from data
            selected_binary_configs=binary_configs,
            selected_meta_config=meta_config,
            sampler=sampler,
            calibrate_meta=effective_config.calibration.enabled != "false",
            calibration_method=effective_config.calibration.method,
            calibration_cv=effective_config.calibration.cv,
            calibration_bins=effective_config.calibration.bins,
            isotonic_min_samples=effective_config.calibration.isotonic_min_samples,
            backend=effective_config.backend,
            model_set=effective_config.model_set,
            device=effective_config.device,
            torch_dtype=effective_config.torch_dtype,
            torch_num_workers=effective_config.torch_num_workers,
            random_state=effective_config.validation.nested_cv.seed,
            max_iter=10000,
            outdir=run_dir,
        )

        # Train final model
        result = train_final_meta_model(final_config)

        if not result.success:
            raise ValueError(
                f"Final model training failed. See {run_dir / 'sanity_checks.json'} for details."
            )

        if result.warnings:
            logger.warning(f"Warnings during training:")
            for w in result.warnings:
                logger.warning(f"  - {w}")

    elif effective_config.task.mode == "binary":
        logger.info("\n[2/4] Training final binary classifier (from scratch)...")
        # Legacy path for binary mode
        metrics_path = technical_run / "metrics_inner_cv.csv"
        best_cfg = _select_best_binary_config(
            metrics_path,
            effective_config.models.selection_metric,
            effective_config.models.selection_direction,
        )
        if sampler:
            best_cfg["sampler"] = sampler
        pos_label = _train_final_binary(effective_config, train_manifest, run_dir, best_cfg)
        task_definitions = {"binary_task": f"positive_class={pos_label}"}
        best_models = {"binary_task": best_cfg["model_name"]}

    elif effective_config.task.mode == "multiclass":
        logger.info("\n[2/4] Training final multiclass model (from scratch)...")
        metrics_path = technical_run / "metrics_inner_cv.csv"
        best_cfg = _select_best_multiclass_config(
            metrics_path,
            effective_config.models.selection_metric,
            effective_config.models.selection_direction,
        )
        if sampler:
            best_cfg["sampler"] = sampler
        _train_final_multiclass(effective_config, train_manifest, run_dir, best_cfg)

    else:
        logger.info("\n[2/4] Training final hierarchical model...")
        metrics_path = technical_run / "metrics_inner_cv.csv"
        _train_final_hierarchical(effective_config, train_manifest, run_dir, metrics_path)

    # Create training manifest
    logger.info("\n[3/4] Creating training manifest...")
    manifest = TrainingRunManifest(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        package_version=__version__,
        training_data_path=str(train_manifest),
        training_data_hash=train_entry.sha256,
        training_data_size_bytes=train_entry.size_bytes,
        training_data_row_count=train_entry.stats.rows,
        config={
            **effective_config.model_dump(mode="python"),
            "final_model": {
                "sampler": sampler,
                "technical_run": str(technical_run),
                "train_from_scratch": True,
            },
        },
        task_type=effective_config.task.mode,
        python_version=platform.python_version(),
        hostname=platform.node(),
        git_hash=_git_hash(paths.root),
        feature_list=train_entry.data_schema.feature_columns,
        task_definitions=task_definitions,
        best_models=best_models,
    )
    manifest.save(run_dir / "run.json")

    # Create bundle
    logger.info("\n[4/4] Creating model bundle...")
    from classiflow.bundles.create import create_bundle

    bundle_path = create_bundle(
        run_dir,
        run_dir / "model_bundle",
        include_metrics=True,
        description=f"Final model trained from scratch on {train_entry.stats.rows} samples",
    )

    # Create lineage record
    lineage = _lineage_payload(
        phase="FINAL_MODEL",
        run_id=run_id,
        config_hash=config_hash,
        train_hash=train_entry.sha256,
        test_hash=test_entry.sha256 if test_entry else None,
        command="classiflow project build-bundle",
        args={
            "run_id": run_id,
            "technical_run": technical_run.name,
            "sampler": sampler,
            "train_from_scratch": True,
        },
        root=paths.root,
        outputs=[bundle_path, run_dir / "run.json", run_dir / "sanity_checks.json"],
    )
    with open(run_dir / "lineage.json", "w", encoding="utf-8") as handle:
        json.dump(lineage, handle, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL MODEL BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Bundle: {bundle_path}")
    logger.info(f"Run directory: {run_dir}")

    return run_dir


def _apply_calibration_selection(config: ProjectConfig, technical_run: Path) -> ProjectConfig:
    if config.task.mode != "meta":
        return config
    comparison_path = technical_run / "calibration_comparison.json"
    if not comparison_path.exists():
        return config
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except Exception:
        return config
    selected = comparison.get("selection", {}).get("method_selected")
    if not selected or selected == config.calibration.method:
        return config
    updated = config.model_copy(deep=True)
    updated.calibration.method = selected
    return updated


def run_independent_test(
    paths: ProjectPaths,
    config: ProjectConfig,
    bundle_path: Path,
    run_id: Optional[str] = None,
) -> Path:
    """
    Run independent test evaluation using the final model bundle.

    This function performs inference on the locked independent test set
    using ONLY the final model bundle. It does NOT use fold pipelines
    or any other artifacts from technical validation.

    The independent test is inference-only - test labels are used only
    for computing metrics, never for model tuning or selection.

    Parameters
    ----------
    paths : ProjectPaths
        Project path configuration
    config : ProjectConfig
        Project configuration
    bundle_path : Path
        Path to the final model bundle (model_bundle.zip)
    run_id : Optional[str]
        Override run ID

    Returns
    -------
    run_dir : Path
        Path to independent test run directory
    """
    logger.info("=" * 70)
    logger.info("INDEPENDENT TEST EVALUATION")
    logger.info("=" * 70)

    registry = _load_registry(paths)
    train_entry = registry.datasets.get("train")
    test_entry = registry.datasets.get("test")
    if not train_entry:
        raise ValueError("Training dataset not registered")
    if not test_entry or config.data.test is None:
        raise ValueError("Test dataset not registered or configured")

    if config.data.test is None:
        raise ValueError("Project has no test manifest configured")
    test_manifest = _resolve_manifest(paths.root, config.data.test.manifest)

    # Verify test dataset hash
    if not verify_manifest_hash(test_manifest, test_entry.sha256):
        raise ValueError(
            "Test manifest hash mismatch. The test dataset has changed since registration. "
            "Re-register with 'classiflow project register-dataset --type test'"
        )

    logger.info(f"Bundle: {bundle_path}")
    logger.info(f"Test manifest: {test_manifest}")
    logger.info(f"Test samples: {test_entry.stats.rows}")

    config_hash = _config_hash(config.model_dump(mode="python"))
    run_id = run_id or _run_id(
        "independent_test", config_hash, train_entry.sha256, test_entry.sha256
    )
    run_dir = paths.runs_subdir("independent_test", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {run_dir}")

    from classiflow.bundles import load_bundle

    # Load bundle and verify it's a final model bundle (not a fold artifact)
    logger.info("\n[1/3] Loading final model bundle...")
    bundle_data = load_bundle(bundle_path, fold=1)

    # Check for final model indicators
    bundle_manifest = bundle_data.get("manifest")
    if bundle_manifest:
        bundle_config = getattr(bundle_manifest, "config", {})
        if isinstance(bundle_config, dict):
            final_model_info = bundle_config.get("final_model", {})
            if final_model_info.get("train_from_scratch"):
                logger.info("  Verified: Bundle trained from scratch on full training data")
            sampler_used = final_model_info.get("sampler", "unknown")
            logger.info(f"  Sampler used: {sampler_used}")

    # Run inference
    logger.info("\n[2/3] Running inference on test set...")
    try:
        infer_config = InferenceConfig(
            run_dir=bundle_data["fold_dir"],
            data_csv=test_manifest,
            output_dir=run_dir,
            id_col=config.key_columns.sample_id,
            label_col=config.key_columns.label,
            include_excel=True,
            include_plots=True,
            verbose=1,
        )
        results = run_inference(infer_config)
    finally:
        bundle_data["loader"].cleanup()

    # Save metrics
    logger.info("\n[3/3] Computing and saving metrics...")
    metrics = results.get("metrics", {})
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            summary_metrics[key] = value
        elif isinstance(value, (int, float)):
            summary_metrics[key] = float(value)
    if "overall" in metrics:
        for key, value in metrics["overall"].items():
            if isinstance(value, bool):
                summary_metrics[key] = value
            elif isinstance(value, (int, float)):
                summary_metrics[key] = float(value)

    # Log key metrics
    overall = metrics.get("overall", {})
    if overall:
        logger.info(f"  Accuracy: {overall.get('accuracy', 'N/A')}")
        logger.info(f"  Balanced Accuracy: {overall.get('balanced_accuracy', 'N/A')}")
        logger.info(f"  F1 (Macro): {overall.get('f1_macro', 'N/A')}")
        if "brier_calibrated" in overall:
            logger.info(f"  Brier Score: {overall.get('brier_calibrated', 'N/A')}")

    test_notes: List[str] = []
    cal_method = overall.get("calibration_method")
    if cal_method:
        test_notes.append(f"Calibration method: {cal_method}")
    if "calibration_enabled" in overall:
        test_notes.append(f"Calibration enabled: {overall.get('calibration_enabled')}")
    calibration_curve_path = run_dir / "calibration_curve.csv"
    if calibration_curve_path.exists():
        test_notes.append(f"Calibration curve exported as {calibration_curve_path.name}")

    # Add note about final model source
    test_notes.append(f"Model bundle: {bundle_path.name}")
    test_notes.append("Model trained from scratch on full training data")

    write_test_report(run_dir / "reports", summary_metrics, notes=test_notes)

    lineage = _lineage_payload(
        phase="INDEPENDENT_TEST",
        run_id=run_id,
        config_hash=config_hash,
        train_hash=train_entry.sha256,
        test_hash=test_entry.sha256,
        command="classiflow project run-test",
        args={
            "run_id": run_id,
            "bundle": str(bundle_path),
            "test_manifest_hash": test_entry.sha256,
        },
        root=paths.root,
        outputs=[metrics_path],
    )
    with open(run_dir / "lineage.json", "w", encoding="utf-8") as handle:
        json.dump(lineage, handle, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("INDEPENDENT TEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results: {run_dir}")

    return run_dir
