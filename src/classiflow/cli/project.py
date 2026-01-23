"""Project-oriented CLI commands."""

from __future__ import annotations

import logging
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

import typer

from classiflow.projects.dataset_registry import register_dataset
from classiflow.projects.orchestrator import (
    run_technical_validation,
    run_feasibility,
    build_final_model,
    run_independent_test,
)
from classiflow.projects.project_fs import ProjectPaths, project_root, choose_project_id
from classiflow.projects.project_models import ProjectConfig, ThresholdsConfig, StabilityGate
from classiflow.projects.reporting import write_promotion_report
from classiflow.projects.promotion import evaluate_promotion, promotion_decision, normalize_metric_name
from classiflow.projects.yaml_utils import dump_yaml

logger = logging.getLogger(__name__)

project_app = typer.Typer(
    name="project",
    help="Project-oriented workflows for clinical test development.",
    add_completion=False,
)


def _latest_run(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_config(paths: ProjectPaths) -> ProjectConfig:
    if not paths.project_yaml.exists():
        raise typer.BadParameter(f"project.yaml not found in {paths.root}")
    return ProjectConfig.load(paths.project_yaml)


def _load_thresholds(paths: ProjectPaths) -> ThresholdsConfig:
    return ThresholdsConfig.load(paths.thresholds_yaml)


def _write_readme(paths: ProjectPaths, project_id: str, name: str) -> None:
    template_path = Path(__file__).resolve().parent.parent / "projects" / "templates" / "README.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        paths.readme.write_text(template.format(project_id=project_id, project_name=name), encoding="utf-8")
    else:
        paths.readme.write_text(f"# {name}\n\nTest ID: `{project_id}`\n", encoding="utf-8")


def _append_torch_backend_hint(project_yaml: Path) -> None:
    """Append commented torch backend example to project.yaml."""
    hint = (
        "\n# Torch backend example (binary/meta)\n"
        "# backend: torch\n"
        "# device: mps\n"
        "# model_set: torch_basic\n"
        "# torch_dtype: float32\n"
        "# torch_num_workers: 0\n"
        "# require_torch_device: false\n"
        "#\n"
        "# Multiclass torch estimators (keep backend: sklearn)\n"
        "# device: mps\n"
        "# multiclass:\n"
        "#   estimator_mode: torch_only\n"
    )
    try:
        with open(project_yaml, "a", encoding="utf-8") as handle:
            handle.write(hint)
    except OSError:
        return


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _infer_key_columns(train_manifest: Path) -> dict:
    df = pd.read_csv(train_manifest, nrows=5)
    columns = df.columns.tolist()
    return {
        "sample_id": _pick_column(columns, ["sample_id", "sample", "id"]),
        "patient_id": _pick_column(columns, ["patient_id", "patient", "subject_id"]),
        "label": _pick_column(columns, ["label", "diagnosis", "target", "class", "subtype", "y"]) or "label",
        "slide_id": _pick_column(columns, ["slide_id", "slide"]),
        "specimen_id": _pick_column(columns, ["specimen_id", "specimen"]),
    }


@project_app.command("init")
def init_project(
    name: str = typer.Option(..., "--name", help="Project short name"),
    test_id: Optional[str] = typer.Option(None, "--test-id", help="Project/test identifier"),
    out_dir: Path = typer.Option(Path("projects"), "--out", help="Base projects directory"),
):
    """Create an empty project scaffold."""
    project_id = choose_project_id(name, test_id)
    root = project_root(out_dir, project_id, name)
    paths = ProjectPaths(root)
    paths.ensure()

    if paths.project_yaml.exists():
        raise typer.BadParameter(f"Project already exists: {paths.project_yaml}")

    config = ProjectConfig(
        project={"id": project_id, "name": name, "description": "", "owner": ""},
        data={
            "train": {"manifest": "data/train/manifest.csv"},
            "test": {"manifest": "data/test/manifest.csv"},
        },
    )
    config.save(paths.project_yaml)
    _append_torch_backend_hint(paths.project_yaml)
    _append_torch_backend_hint(paths.project_yaml)

    thresholds = ThresholdsConfig()
    thresholds.save(paths.thresholds_yaml)

    dump_yaml({"labels": {}}, paths.labels_yaml)
    dump_yaml({"features": {}}, paths.features_yaml)

    _write_readme(paths, project_id, name)

    typer.echo(str(paths.root))


@project_app.command("bootstrap")
def bootstrap_project(
    train_manifest: Path = typer.Option(..., "--train-manifest", help="Training manifest path"),
    test_manifest: Optional[Path] = typer.Option(None, "--test-manifest", help="Test manifest path"),
    name: str = typer.Option(..., "--name", help="Project short name"),
    out_dir: Path = typer.Option(Path("projects"), "--out", help="Base projects directory"),
    mode: str = typer.Option("auto", "--mode", help="Task mode: auto|binary|meta|multiclass|hierarchical"),
    hierarchy: Optional[str] = typer.Option(None, "--hierarchy", help="Hierarchy column/path"),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Label column name"),
    sample_id_col: Optional[str] = typer.Option(None, "--sample-id-col", help="Sample ID column name"),
    patient_id_col: Optional[str] = typer.Option(
        None,
        "--patient-id-col",
        "--patient-col",
        help="Patient ID column name",
    ),
    no_patient_stratified: bool = typer.Option(False, "--no-patient-stratified", help="Disable patient stratification"),
    thresholds: List[str] = typer.Option([], "--threshold", help="Metric threshold overrides (metric:value)"),
    copy_data: str = typer.Option("pointer", "--copy-data", help="copy|symlink|pointer"),
    test_id: Optional[str] = typer.Option(None, "--test-id", help="Project/test identifier"),
):
    """Bootstrap a project with dataset registration."""
    project_id = choose_project_id(name, test_id)
    root = project_root(out_dir, project_id, name)
    paths = ProjectPaths(root)
    paths.ensure()

    if copy_data not in {"copy", "symlink", "pointer"}:
        raise typer.BadParameter("copy-data must be copy, symlink, or pointer")

    train_dest = train_manifest
    test_dest = test_manifest

    if copy_data != "pointer":
        train_dest = paths.data_dir / "train" / train_manifest.name
        if test_manifest is not None:
            test_dest = paths.data_dir / "test" / test_manifest.name
        train_dest.parent.mkdir(parents=True, exist_ok=True)
        if test_manifest is not None:
            test_dest.parent.mkdir(parents=True, exist_ok=True)
        if copy_data == "copy":
            shutil.copyfile(train_manifest, train_dest)
            if test_manifest is not None:
                shutil.copyfile(test_manifest, test_dest)
        else:
            if not train_dest.exists():
                train_dest.symlink_to(train_manifest)
            if test_manifest is not None and not test_dest.exists():
                test_dest.symlink_to(test_manifest)
    else:
        train_dest = train_manifest.resolve()
        if test_manifest is not None:
            test_dest = test_manifest.resolve()

    key_columns = _infer_key_columns(train_dest)
    if label_col:
        key_columns["label"] = label_col
    if sample_id_col:
        key_columns["sample_id"] = sample_id_col
    if patient_id_col:
        key_columns["patient_id"] = patient_id_col
    inferred_mode = "meta"
    if mode == "auto":
        label_col = key_columns["label"]
        try:
            label_series = pd.read_csv(train_dest, usecols=[label_col])[label_col]
            n_labels = label_series.nunique()
            inferred_mode = "binary" if n_labels <= 2 else "meta"
        except Exception:
            inferred_mode = "meta"
    data_config = {
        "train": {"manifest": str(train_dest)},
        "test": {"manifest": str(test_dest)} if test_dest is not None else None,
    }
    config = ProjectConfig(
        project={"id": project_id, "name": name, "description": "", "owner": ""},
        data=data_config,
        key_columns=key_columns,
        task={
            "mode": inferred_mode if mode == "auto" else mode,
            "patient_stratified": not no_patient_stratified,
            "hierarchy_path": hierarchy,
        },
    )

    config.save(paths.project_yaml)
    _append_torch_backend_hint(paths.project_yaml)

    thresholds_cfg = ThresholdsConfig()
    thresholds_cfg.technical_validation.required = {
        "f1": 0.7,
        "balanced_accuracy": 0.7,
    }
    thresholds_cfg.technical_validation.stability = StabilityGate(
        std_max={"f1": 0.1, "balanced_accuracy": 0.1},
        pass_rate_min=0.8,
    )
    thresholds_cfg.independent_test.required = {
        "f1_macro": 0.7,
        "balanced_accuracy": 0.7,
    }
    thresholds_cfg.promotion.calibration.brier_max = 0.20
    thresholds_cfg.promotion.calibration.ece_max = 0.25
    for entry in thresholds:
        if ":" not in entry:
            continue
        metric, value = entry.split(":", 1)
        try:
            thresholds_cfg.technical_validation.required[metric] = float(value)
        except ValueError:
            raise typer.BadParameter(f"Invalid threshold: {entry}")
    thresholds_cfg.save(paths.thresholds_yaml)

    register_dataset(paths.datasets_yaml, config, "train", train_dest)
    if test_dest is not None:
        register_dataset(paths.datasets_yaml, config, "test", test_dest)

    _write_readme(paths, project_id, name)

    typer.echo(str(paths.root))


@project_app.command("register-dataset")
def register_dataset_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    dataset_type: str = typer.Option(..., "--type", help="train or test"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest path override"),
):
    """Register or re-register dataset manifests."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)

    manifest_path = manifest
    if manifest_path is None:
        if dataset_type == "train":
            raw_path = config.data.train.manifest
        else:
            if config.data.test is None:
                raise typer.BadParameter("Project has no test manifest configured")
            raw_path = config.data.test.manifest
        manifest_path = Path(raw_path)
        if not manifest_path.is_absolute():
            manifest_path = project_dir / manifest_path
    entry = register_dataset(paths.datasets_yaml, config, dataset_type, manifest_path)
    typer.echo(f"Registered {dataset_type} dataset: {entry.sha256[:12]}...")


@project_app.command("run-technical")
def run_technical_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run id"),
    compare_smote: bool = typer.Option(False, "--compare-smote", help="Run SMOTE comparison"),
    tracker: Optional[str] = typer.Option(
        None,
        "--tracker",
        help="Experiment tracking backend: mlflow, wandb (requires optional deps)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Experiment/project name for tracking (default: {project_id}/technical)",
    ),
):
    """Run technical validation."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)
    # Override tracking settings from CLI if provided
    if tracker:
        config.tracker = tracker
    if experiment_name:
        config.experiment_name = experiment_name
    run_dir = run_technical_validation(paths, config, run_id=run_id, compare_smote=compare_smote)
    typer.echo(str(run_dir))


@project_app.command("run-feasibility")
def run_feasibility_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run id"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Subset/order of classes"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance threshold"),
    min_n: int = typer.Option(3, "--min-n", help="Minimum n per class for Shapiro-Wilk"),
    dunn_adjust: str = typer.Option("holm", "--dunn-adjust", help="P-value adjustment for Dunn test"),
    top_n: int = typer.Option(30, "--top-n", help="Number of top features in summary"),
    no_legacy_csv: bool = typer.Option(False, "--no-legacy-csv", help="Skip legacy CSV outputs"),
    no_legacy_xlsx: bool = typer.Option(False, "--no-legacy-xlsx", help="Skip legacy xlsx output"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip visualization outputs"),
    fc_thresh: float = typer.Option(1.0, "--fc-thresh", help="|log2FC| threshold for volcano"),
    fc_center: str = typer.Option("median", "--fc-center", help="Center for fold-change (mean/median)"),
    label_topk: int = typer.Option(12, "--label-topk", help="Top features to annotate on volcano"),
    heatmap_topn: int = typer.Option(30, "--heatmap-topn", help="Top features for heatmap (0=skip)"),
    fig_dpi: int = typer.Option(160, "--fig-dpi", help="Figure DPI"),
):
    """Run feasibility stats + visualizations."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)
    run_dir = run_feasibility(
        paths,
        config,
        run_id=run_id,
        classes=classes,
        alpha=alpha,
        min_n=min_n,
        dunn_adjust=dunn_adjust,
        top_n=top_n,
        write_legacy_csv=not no_legacy_csv,
        write_legacy_xlsx=not no_legacy_xlsx,
        run_viz=not no_viz,
        fc_thresh=fc_thresh,
        fc_center=fc_center,
        label_topk=label_topk,
        heatmap_topn=heatmap_topn,
        fig_dpi=fig_dpi,
    )
    typer.echo(str(run_dir))


@project_app.command("build-bundle")
def build_bundle_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    technical_run_id: Optional[str] = typer.Option(None, "--technical-run", help="Technical run id"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run id"),
    sampler: Optional[str] = typer.Option(
        None,
        "--sampler",
        help="Sampler for final training: none, smote (auto-selects if not specified)",
    ),
):
    """
    Train final model and build bundle.

    The final model is ALWAYS trained from scratch on 100% of the training data
    using the validated per-task configurations from technical validation.

    This ensures:
    - No reuse of fold pipelines (deterministic, auditable training)
    - Per-task hyperparameter selection (not global selection)
    - Sanity checks to detect degenerate predictions
    - Complete artifact trail for clinical-grade reproducibility

    The --sampler option controls whether SMOTE is applied during final training:
    - none: No oversampling (default if SMOTE disabled in config)
    - smote: Apply SMOTE oversampling
    - If not specified, auto-selects based on config and technical validation results
    """
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)

    technical_root = paths.runs_dir / "technical_validation"
    technical_run = technical_root / technical_run_id if technical_run_id else _latest_run(technical_root)
    if not technical_run:
        raise typer.BadParameter("No technical validation runs found")

    if sampler and sampler not in ("none", "smote"):
        raise typer.BadParameter(f"Invalid sampler: {sampler}. Must be 'none' or 'smote'")

    run_dir = build_final_model(paths, config, technical_run, run_id=run_id, sampler=sampler)
    typer.echo(str(run_dir))


@project_app.command("run-test")
def run_test_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    bundle: Optional[Path] = typer.Option(None, "--bundle", help="Bundle ZIP path"),
    final_run_id: Optional[str] = typer.Option(None, "--final-run", help="Final model run id"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run id"),
):
    """Run independent test evaluation."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)

    if bundle is None:
        final_root = paths.runs_dir / "final_model"
        final_run = final_root / final_run_id if final_run_id else _latest_run(final_root)
        if not final_run:
            raise typer.BadParameter("No final model runs found")
        bundle = final_run / "model_bundle.zip"

    run_dir = run_independent_test(paths, config, bundle, run_id=run_id)
    typer.echo(str(run_dir))


@project_app.command("recommend")
def recommend_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    technical_run_id: Optional[str] = typer.Option(None, "--technical-run", help="Technical run id"),
    test_run_id: Optional[str] = typer.Option(None, "--test-run", help="Independent test run id"),
    override: bool = typer.Option(False, "--override", help="Override failed gates"),
    comment: Optional[str] = typer.Option(None, "--comment", help="Override comment"),
    approver: Optional[str] = typer.Option(None, "--approver", help="Override approver"),
):
    """Evaluate promotion gates and emit recommendation."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)
    thresholds = _load_thresholds(paths)

    technical_root = paths.runs_dir / "technical_validation"
    test_root = paths.runs_dir / "independent_test"

    technical_run = technical_root / technical_run_id if technical_run_id else _latest_run(technical_root)
    test_run = test_root / test_run_id if test_run_id else _latest_run(test_root)

    if not technical_run or not test_run:
        raise typer.BadParameter("Both technical and independent test runs are required")

    tech_metrics_path = technical_run / "metrics_summary.json"
    test_metrics_path = test_run / "metrics.json"

    if not tech_metrics_path.exists() or not test_metrics_path.exists():
        raise typer.BadParameter("Missing metrics files for recommendation")

    tech_payload = json.loads(tech_metrics_path.read_text(encoding="utf-8"))
    test_payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))

    tech_summary = tech_payload.get("summary", {})
    tech_per_fold = tech_payload.get("per_fold", {})
    test_summary = {}
    if "overall" in test_payload:
        for key, value in test_payload["overall"].items():
            if isinstance(value, (int, float)):
                test_summary[key] = float(value)

    gate_results = evaluate_promotion(thresholds, tech_summary, tech_per_fold, test_summary)
    decision, reasons = promotion_decision(gate_results)

    if override and thresholds.override.allow_override:
        if thresholds.override.require_comment and not comment:
            raise typer.BadParameter("Override comment required")
        if thresholds.override.require_approver and not approver:
            raise typer.BadParameter("Override approver required")
        decision = True
        reasons.append("Override applied")

    gate_rows = []
    for phase, result in gate_results.items():
        gate_rows.append({
            "phase": phase,
            "passed": result.passed,
            "reasons": "; ".join(result.reasons) if result.reasons else "",
        })

    gating_rows = []
    report_only_rows = []
    cal_thresholds = thresholds.promotion.calibration
    phases = {
        "technical_validation": (tech_summary, thresholds.technical_validation),
        "independent_test": (test_summary, thresholds.independent_test),
    }
    for phase, (metrics, phase_thresholds) in phases.items():
        for metric_name, threshold in phase_thresholds.required.items():
            normalized = normalize_metric_name(metric_name)
            actual = metrics.get(normalized)
            passed = actual is not None and actual >= threshold
            gating_rows.append({
                "phase": phase,
                "metric": metric_name,
                "value": actual,
                "threshold": threshold,
                "direction": ">=",
                "passed": passed,
            })
        for metric_name, max_allowed in phase_thresholds.safety.items():
            normalized = normalize_metric_name(metric_name)
            actual = metrics.get(normalized)
            passed = actual is not None and actual <= max_allowed
            gating_rows.append({
                "phase": phase,
                "metric": metric_name,
                "value": actual,
                "threshold": max_allowed,
                "direction": "<=",
                "passed": passed,
            })
        brier_value = metrics.get("brier_calibrated")
        gating_rows.append({
            "phase": phase,
            "metric": "brier_calibrated",
            "value": brier_value,
            "threshold": cal_thresholds.brier_max,
            "direction": "<=",
            "passed": brier_value is not None and brier_value <= cal_thresholds.brier_max,
        })
        ece_value = metrics.get("ece_calibrated")
        gating_rows.append({
            "phase": phase,
            "metric": "ece_calibrated",
            "value": ece_value,
            "threshold": cal_thresholds.ece_max,
            "direction": "<=",
            "passed": ece_value is not None and ece_value <= cal_thresholds.ece_max,
        })
        if phase == "technical_validation" and phase_thresholds.stability:
            stability = phase_thresholds.stability
            for metric_name, max_std in stability.std_max.items():
                normalized = normalize_metric_name(metric_name)
                actual = gate_results["technical_validation"].metrics.get(f"{normalized}_std")
                passed = actual is not None and actual <= max_std
                gating_rows.append({
                    "phase": phase,
                    "metric": f"{metric_name}_std",
                    "value": actual,
                    "threshold": max_std,
                    "direction": "<=",
                    "passed": passed,
                })
            for metric_name in phase_thresholds.required.keys():
                normalized = normalize_metric_name(metric_name)
                actual = gate_results["technical_validation"].metrics.get(f"{normalized}_pass_rate")
                passed = actual is not None and actual >= stability.pass_rate_min
                gating_rows.append({
                    "phase": phase,
                    "metric": f"{metric_name}_pass_rate",
                    "value": actual,
                    "threshold": stability.pass_rate_min,
                    "direction": ">=",
                    "passed": passed,
                })

        report_only_keys = {
            "roc_auc",
            "roc_auc_ovr_macro",
            "roc_auc_macro",
            "pr_auc",
            "accuracy",
            "log_loss",
            "brier",
            "ece",
            "log_loss_uncalibrated",
            "brier_uncalibrated",
            "ece_uncalibrated",
            "calibration_method",
            "calibration_enabled",
            "calibration_bins",
        }
        for key, value in metrics.items():
            if key in report_only_keys:
                report_only_rows.append({
                    "phase": phase,
                    "metric": key,
                    "value": value,
                })

    calibration_selection = {}
    comparison_path = technical_run / "calibration_comparison.json"
    if comparison_path.exists():
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            selection = comparison.get("selection", {})
            if selection:
                calibration_selection = selection
        except Exception:
            calibration_selection = {}

    report_path = write_promotion_report(
        paths.promotion_dir,
        decision,
        reasons,
        pd.DataFrame(gate_rows),
        pd.DataFrame(gating_rows),
        pd.DataFrame(report_only_rows),
        calibration_selection=calibration_selection,
    )

    decision_payload = {
        "decision": "PASS" if decision else "FAIL",
        "timestamp": datetime.utcnow().isoformat(),
        "technical_run": technical_run.name,
        "test_run": test_run.name,
        "reasons": reasons,
        "override": {
            "enabled": override,
            "comment": comment,
            "approver": approver,
        },
    }
    dump_yaml(decision_payload, paths.promotion_dir / "decision.yaml")
    decision_json = {
        **decision_payload,
        "gates": gate_rows,
        "metrics": {
            "gating": gating_rows,
            "report_only": report_only_rows,
        },
        "calibration": calibration_selection,
    }
    with open(paths.promotion_dir / "promotion_decision.json", "w", encoding="utf-8") as handle:
        json.dump(decision_json, handle, indent=2)
    typer.echo(str(report_path))


@project_app.command("ship")
def ship_cmd(
    project_dir: Path = typer.Argument(..., help="Project root directory"),
    out_dir: Path = typer.Option(..., "--out", help="Output directory for deployment bundle"),
    bundle: Optional[Path] = typer.Option(None, "--bundle", help="Bundle ZIP path override"),
    final_run_id: Optional[str] = typer.Option(None, "--final-run", help="Final model run id"),
    include_promotion: bool = typer.Option(True, "--include-promotion/--no-promotion", help="Copy promotion decision"),
):
    """Copy the exact model bundle + metadata for deployment."""
    paths = ProjectPaths(project_dir)
    config = _load_config(paths)

    if bundle is None:
        final_root = paths.runs_dir / "final_model"
        final_run = final_root / final_run_id if final_run_id else _latest_run(final_root)
        if not final_run:
            raise typer.BadParameter("No final model runs found")
        bundle = final_run / "model_bundle.zip"
        if not bundle.exists():
            raise typer.BadParameter(f"Bundle not found: {bundle}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shipped_bundle = out_dir / bundle.name
    shutil.copy2(bundle, shipped_bundle)

    metadata = {
        "project_id": config.project.id,
        "project_name": config.project.name,
        "bundle": str(shipped_bundle),
        "shipped_at": datetime.utcnow().isoformat(),
    }

    run_manifest = bundle.parent / "run.json"
    if run_manifest.exists():
        shutil.copy2(run_manifest, out_dir / "run.json")
        metadata["run_manifest"] = str(out_dir / "run.json")

    lineage = bundle.parent / "lineage.json"
    if lineage.exists():
        shutil.copy2(lineage, out_dir / "lineage.json")
        metadata["lineage"] = str(out_dir / "lineage.json")

    if include_promotion:
        decision = paths.promotion_dir / "decision.yaml"
        if decision.exists():
            shutil.copy2(decision, out_dir / "decision.yaml")
            metadata["promotion_decision"] = str(out_dir / "decision.yaml")

    dump_yaml(metadata, out_dir / "ship_manifest.yaml")
    typer.echo(str(out_dir))
