"""Main CLI entrypoint using Typer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List
import sys

import numpy as np
import pandas as pd
import typer

from classiflow import __version__
from classiflow.config import TrainConfig, MetaConfig, MulticlassConfig, HierarchicalConfig, _resolve_data_path
from classiflow.training import train_binary_task, train_meta_classifier, train_multiclass_classifier
from classiflow.io.compatibility import assess_data_compatibility
from classiflow.evaluation.smote_comparison import SMOTEComparison
from classiflow.cli.stats import stats_app
from classiflow.cli.bundle import bundle_app
from classiflow.cli.migrate import migrate_app
from classiflow.cli.project import project_app
from classiflow.cli.config import config_app
from classiflow.cli.ui import ui_app
from classiflow.cli.backfill import backfill_app

app = typer.Typer(
    name="classiflow",
    help="Production-grade ML toolkit for molecular subtype classification.",
    add_completion=False,
)

# Add subcommands
app.add_typer(stats_app, name="stats")
app.add_typer(bundle_app, name="bundle")
app.add_typer(migrate_app, name="migrate")
app.add_typer(project_app, name="project")
app.add_typer(config_app, name="config")
app.add_typer(ui_app, name="ui")
app.add_typer(backfill_app, name="backfill")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"classiflow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """classiflow: ML toolkit for molecular subtype classification."""
    pass


@app.command()
def train_binary(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset). "
        "Recommended format: .parquet for performance and schema stability.",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features + labels. Use --data instead.",
    ),
    patient_col: Optional[str] = typer.Option(
        None,
        "--patient-col",
        help="Column with patient/slide IDs for stratification (optional; if not provided, sample-level stratification is used)",
    ),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column"),
    pos_label: Optional[str] = typer.Option(None, "--pos-label", help="Positive class label (default: minority)"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory"),
    outer_folds: int = typer.Option(3, "--outer-folds", help="Number of outer CV folds"),
    inner_splits: int = typer.Option(5, "--inner-splits", help="Number of inner CV splits"),
    inner_repeats: int = typer.Option(2, "--inner-repeats", help="Number of inner CV repeats"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
    smote: str = typer.Option("off", "--smote", help="SMOTE mode: off, on, both"),
    max_iter: int = typer.Option(10000, "--max-iter", help="Max iterations for linear models"),
    backend: str = typer.Option(
        "sklearn",
        "--backend",
        help="Estimator backend. sklearn (CPU) or torch (GPU-ready via CUDA/MPS).",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, mps",
    ),
    model_set: Optional[str] = typer.Option(
        None,
        "--model-set",
        help="Model set registry key (e.g., torch_basic, torch_fast).",
    ),
    torch_num_workers: int = typer.Option(
        0,
        "--torch-num-workers",
        help="PyTorch DataLoader worker count (torch backend only).",
    ),
    torch_dtype: str = typer.Option(
        "float32",
        "--torch-dtype",
        help="Torch dtype: float32 or float16 (torch backend only).",
    ),
    require_device: bool = typer.Option(
        False,
        "--require-device/--allow-device-fallback",
        help="Require requested torch device (mps/cuda) instead of falling back to CPU.",
    ),
    tracker: Optional[str] = typer.Option(
        None,
        "--tracker",
        help="Experiment tracking backend: mlflow, wandb (requires optional deps)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Experiment/project name for tracking (default: classiflow-binary)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name for tracking (default: auto-generated)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """
    Train a binary classifier with nested cross-validation.

    Supports CSV, Parquet, and Parquet dataset directories.
    Patient-level stratification (optional): Provide --patient-col patient_id to ensure no data leakage by patient across folds.
    If omitted, sample-level stratification is used.

    Examples:
        # Single parquet file (recommended)
        classiflow train-binary --data data.parquet --label-col diagnosis --smote on

        # Dataset directory (chunked parquet)
        classiflow train-binary --data data_parquet/ --label-col diagnosis --smote on

        # Patient-level stratification
        classiflow train-binary --data data.parquet --patient-col patient_id --label-col diagnosis --smote on

        # Legacy CSV (deprecated)
        classiflow train-binary --data-csv data.csv --label-col diagnosis --smote on
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    backend = backend.lower()
    device = device.lower()
    if model_set is None:
        model_set = "torch_basic" if backend == "torch" else "default"

    config = TrainConfig(
        data_path=resolved_path,
        patient_col=patient_col,
        label_col=label_col,
        pos_label=pos_label,
        outdir=outdir,
        outer_folds=outer_folds,
        inner_splits=inner_splits,
        inner_repeats=inner_repeats,
        random_state=random_state,
        smote_mode=smote,
        max_iter=max_iter,
        backend=backend,
        device=device,
        model_set=model_set,
        torch_num_workers=torch_num_workers,
        torch_dtype=torch_dtype,
        require_torch_device=require_device,
        tracker=tracker,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    try:
        results = train_binary_task(config)
        typer.secho(f"\n✓ Training complete. Results saved to {outdir}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"\n✗ Training failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def train_meta(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset). "
        "Recommended format: .parquet for performance and schema stability.",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features + labels. Use --data instead.",
    ),
    patient_col: Optional[str] = typer.Option(
        None,
        "--patient-col",
        help="Column with patient/slide IDs for stratification (optional; if not provided, sample-level stratification is used)",
    ),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Class labels to include (order matters)"),
    tasks_json: Optional[Path] = typer.Option(None, "--tasks-json", help="Optional JSON with composite tasks"),
    tasks_only: bool = typer.Option(False, "--tasks-only", help="If set, only use tasks from JSON (skip auto OvR/pairwise)"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory"),
    outer_folds: int = typer.Option(3, "--outer-folds", help="Number of outer CV folds"),
    inner_splits: int = typer.Option(5, "--inner-splits", help="Number of inner CV splits"),
    inner_repeats: int = typer.Option(2, "--inner-repeats", help="Number of inner CV repeats"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
    smote: str = typer.Option("both", "--smote", help="SMOTE mode: off, on, both"),
    max_iter: int = typer.Option(10000, "--max-iter", help="Max iterations for linear models"),
    backend: str = typer.Option(
        "sklearn",
        "--backend",
        help="Estimator backend. sklearn (CPU) or torch (GPU-ready via CUDA/MPS).",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cpu, cuda, mps",
    ),
    model_set: Optional[str] = typer.Option(
        None,
        "--model-set",
        help="Model set registry key (e.g., torch_basic, torch_fast).",
    ),
    torch_num_workers: int = typer.Option(
        0,
        "--torch-num-workers",
        help="PyTorch DataLoader worker count (torch backend only).",
    ),
    torch_dtype: str = typer.Option(
        "float32",
        "--torch-dtype",
        help="Torch dtype: float32 or float16 (torch backend only).",
    ),
    require_device: bool = typer.Option(
        False,
        "--require-device/--allow-device-fallback",
        help="Require requested torch device (mps/cuda) instead of falling back to CPU.",
    ),
    calibrate_meta: bool = typer.Option(
        True,
        "--calibrate-meta/--no-calibrate-meta",
        help="Enable probability calibration for the meta-classifier.",
    ),
    calibration_method: str = typer.Option(
        "sigmoid",
        "--calibration-method",
        help="Calibration method: sigmoid (default) or isotonic.",
    ),
    calibration_cv: int = typer.Option(
        3,
        "--calibration-cv",
        help="Number of folds for cross-validated calibration.",
    ),
    calibration_bins: int = typer.Option(
        10,
        "--calibration-bins",
        help="Number of bins when computing calibration curves.",
    ),
    calibration_isotonic_min_samples: int = typer.Option(
        100,
        "--calibration-isotonic-min-samples",
        help="Minimum samples to allow isotonic calibration.",
    ),
    tracker: Optional[str] = typer.Option(
        None,
        "--tracker",
        help="Experiment tracking backend: mlflow, wandb (requires optional deps)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Experiment/project name for tracking (default: classiflow-meta)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name for tracking (default: auto-generated)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """
    Train meta-classifier for multiclass problems via binary tasks.

    Automatically builds OvR and pairwise tasks, with optional composite tasks from JSON.
    Use --tasks-only to train ONLY the tasks from JSON (skipping auto OvR/pairwise).

    Supports CSV, Parquet, and Parquet dataset directories.
    Patient-level stratification (optional): Provide --patient-col patient_id to ensure no data leakage by patient across folds.
    If omitted, sample-level stratification is used.

    Examples:
        # Single parquet file (recommended)
        classiflow train-meta --data data.parquet --label-col subtype --smote both

        # Dataset directory (chunked parquet)
        classiflow train-meta --data data_parquet/ --label-col subtype --smote both

        # Patient-level stratification
        classiflow train-meta --data data.parquet --patient-col patient_id --label-col subtype --smote both

        # Legacy CSV (deprecated)
        classiflow train-meta --data-csv data.csv --label-col subtype --tasks-json tasks.json --smote both

        # Only custom tasks from JSON
        classiflow train-meta --data data.parquet --label-col subtype --tasks-json tasks.json --tasks-only --smote both
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    backend = backend.lower()
    device = device.lower()
    if model_set is None:
        model_set = "torch_basic" if backend == "torch" else "default"

    calibration_method = calibration_method.lower().strip()
    valid_methods = {"sigmoid", "isotonic"}
    if calibration_method not in valid_methods:
        raise typer.BadParameter(
            f"Unsupported calibration method '{calibration_method}'. "
            f"Choose from {', '.join(sorted(valid_methods))}."
        )

    config = MetaConfig(
        data_path=resolved_path,
        patient_col=patient_col,
        label_col=label_col,
        classes=classes,
        tasks_json=tasks_json,
        tasks_only=tasks_only,
        outdir=outdir,
        outer_folds=outer_folds,
        inner_splits=inner_splits,
        inner_repeats=inner_repeats,
        random_state=random_state,
        smote_mode=smote,
        max_iter=max_iter,
        backend=backend,
        device=device,
        model_set=model_set,
        torch_num_workers=torch_num_workers,
        torch_dtype=torch_dtype,
        require_torch_device=require_device,
        calibrate_meta=calibrate_meta,
        calibration_enabled="true" if calibrate_meta else "false",
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        calibration_bins=calibration_bins,
        calibration_isotonic_min_samples=calibration_isotonic_min_samples,
        tracker=tracker,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # Check data compatibility before training
    typer.echo("Checking data compatibility...")
    compat_result = assess_data_compatibility(config, return_details=True)
    print(compat_result)

    if not compat_result.is_compatible:
        typer.secho("\n✗ Data is not compatible with train-meta mode.", fg=typer.colors.RED, err=True)
        typer.echo("\nPlease fix the issues above before training.")
        raise typer.Exit(code=1)

    if compat_result.warnings:
        typer.echo("\n⚠️  Proceeding with warnings (see above)")
        if not typer.confirm("Continue with training?", default=True):
            typer.echo("Training cancelled.")
            raise typer.Exit(code=0)

    try:
        results = train_meta_classifier(config)
        typer.secho(f"\n✓ Training complete. Results saved to {outdir}", fg=typer.colors.GREEN)
        typer.echo(f"  Tasks trained: {results['n_tasks']}")
        typer.echo(f"  Folds: {results['n_folds']}")
    except Exception as e:
        typer.secho(f"\n✗ Training failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="train-multiclass")
def train_multiclass(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset). "
        "Recommended format: .parquet for performance and schema stability.",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features + labels. Use --data instead.",
    ),
    patient_col: Optional[str] = typer.Option(
        None,
        "--patient-col",
        help="Column with patient/slide IDs for stratification (optional; if not provided, sample-level stratification is used)",
    ),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Class labels to include (order matters)"),
    outdir: Path = typer.Option(Path("derived"), "--outdir", help="Output directory"),
    outer_folds: int = typer.Option(3, "--outer-folds", help="Number of outer CV folds"),
    inner_splits: int = typer.Option(5, "--inner-splits", help="Number of inner CV splits"),
    inner_repeats: int = typer.Option(2, "--inner-repeats", help="Number of inner CV repeats"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
    smote: str = typer.Option("both", "--smote", help="SMOTE mode: off, on, both"),
    max_iter: int = typer.Option(10000, "--max-iter", help="Max iterations for non-logreg linear models"),
    group_stratify: bool = typer.Option(
        True,
        "--group-stratify/--no-group-stratify",
        help="Use stratified group splits when patient_col is provided",
    ),
    logreg_solver: str = typer.Option("saga", "--logreg-solver", help="LogisticRegression solver"),
    logreg_multi_class: str = typer.Option("auto", "--logreg-multi-class", help="LogisticRegression multi_class"),
    logreg_penalty: str = typer.Option("l2", "--logreg-penalty", help="LogisticRegression penalty"),
    logreg_max_iter: int = typer.Option(5000, "--logreg-max-iter", help="LogisticRegression max_iter"),
    logreg_tol: float = typer.Option(1e-3, "--logreg-tol", help="LogisticRegression tolerance"),
    logreg_C: float = typer.Option(1.0, "--logreg-C", help="LogisticRegression C"),
    logreg_class_weight: Optional[str] = typer.Option(
        "balanced",
        "--logreg-class-weight",
        help="LogisticRegression class_weight (use 'none' for None)",
    ),
    logreg_n_jobs: int = typer.Option(-1, "--logreg-n-jobs", help="LogisticRegression n_jobs"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
    estimator_mode: str = typer.Option(
        "all",
        "--estimator-mode",
        help="Estimator selection: all, torch_only, cpu_only",
    ),
    tracker: Optional[str] = typer.Option(
        None,
        "--tracker",
        help="Experiment tracking backend: mlflow, wandb (requires optional deps)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Experiment/project name for tracking (default: classiflow-multiclass)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name for tracking (default: auto-generated)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """
    Train a multiclass classifier with nested cross-validation.

    Supports CSV, Parquet, and Parquet dataset directories.
    Patient-level stratification (optional): Provide --patient-col patient_id to ensure no data leakage by patient across folds.
    If omitted, sample-level stratification is used.

    Examples:
        # Single parquet file (recommended)
        classiflow train-multiclass --data data.parquet --label-col subtype --smote both

        # Dataset directory (chunked parquet)
        classiflow train-multiclass --data data_parquet/ --label-col subtype --smote both

        # Patient-level stratification
        classiflow train-multiclass --data data.parquet --patient-col patient_id --label-col subtype --smote both

        # Legacy CSV (deprecated)
        classiflow train-multiclass --data-csv data.csv --label-col subtype --smote both
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    normalized_class_weight = logreg_class_weight
    if isinstance(normalized_class_weight, str) and normalized_class_weight.lower() in {"none", "null"}:
        normalized_class_weight = None

    config = MulticlassConfig(
        data_path=resolved_path,
        patient_col=patient_col,
        label_col=label_col,
        classes=classes,
        outdir=outdir,
        outer_folds=outer_folds,
        inner_splits=inner_splits,
        inner_repeats=inner_repeats,
        random_state=random_state,
        smote_mode=smote,
        max_iter=max_iter,
        group_stratify=group_stratify,
        logreg_solver=logreg_solver,
        logreg_multi_class=logreg_multi_class,
        # penalty deprecated in sklearn 1.8; keep default and tune via l1_ratio/C
        logreg_max_iter=logreg_max_iter,
        logreg_tol=logreg_tol,
        logreg_C=logreg_C,
        logreg_class_weight=normalized_class_weight,
        logreg_n_jobs=logreg_n_jobs,
        device=device,
        estimator_mode=estimator_mode,
        tracker=tracker,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    try:
        train_multiclass_classifier(config)
        typer.secho(f"\n✓ Training complete. Results saved to {outdir}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"\n✗ Training failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="check-compatibility")
def check_compatibility(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset).",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features + labels. Use --data instead.",
    ),
    mode: str = typer.Option(..., "--mode", help="Training mode: meta or hierarchical"),
    label_col: str = typer.Option(..., "--label-col", help="Name of label column (or L1 for hierarchical)"),
    label_l2: Optional[str] = typer.Option(None, "--label-l2", help="Level-2 label column (hierarchical only)"),
    patient_col: Optional[str] = typer.Option(None, "--patient-col", help="Patient ID column for stratification (hierarchical only, optional)"),
    classes: Optional[List[str]] = typer.Option(None, "--classes", help="Class labels to include (meta only)"),
    outer_folds: int = typer.Option(3, "--outer-folds", help="Number of outer CV folds"),
):
    """
    Check if data is compatible with classiflow training modes.

    This command assesses input data without running training, providing detailed
    feedback about compatibility issues, warnings, and suggestions for fixes.

    Supports CSV, Parquet, and Parquet dataset directories.

    Examples:
        # Check meta-classifier compatibility (parquet)
        classiflow check-compatibility --data data.parquet --mode meta --label-col diagnosis

        # Check hierarchical compatibility (without patient stratification)
        classiflow check-compatibility --data data.parquet --mode hierarchical --label-col tumor_type --label-l2 subtype

        # Check hierarchical compatibility (with patient stratification)
        classiflow check-compatibility --data data.parquet --mode hierarchical --label-col tumor_type --label-l2 subtype --patient-col patient_id

        # Legacy CSV (deprecated)
        classiflow check-compatibility --data-csv data.csv --mode meta --label-col diagnosis --outer-folds 5
    """
    if mode not in ["meta", "hierarchical"]:
        typer.secho(f"Error: Invalid mode '{mode}'. Must be 'meta' or 'hierarchical'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create config based on mode
        if mode == "meta":
            config = MetaConfig(
                data_path=resolved_path,
                label_col=label_col,
                classes=classes,
                outer_folds=outer_folds,
            )
        else:  # hierarchical
            config = HierarchicalConfig(
                data_path=resolved_path,
                patient_col=patient_col,
                label_l1=label_col,
                label_l2=label_l2,
                outer_folds=outer_folds,
            )

        # Assess compatibility
        result = assess_data_compatibility(config, return_details=True)

        # Print results
        print(result)

        # Exit with appropriate code
        if not result.is_compatible:
            raise typer.Exit(code=1)
        elif result.warnings:
            typer.echo("\n✓ Data is compatible but has warnings (see above)")
            raise typer.Exit(code=0)
        else:
            typer.secho("\n✓ Data is fully compatible!", fg=typer.colors.GREEN)
            raise typer.Exit(code=0)

    except typer.Exit:
        # Re-raise typer.Exit without catching it as an error
        raise
    except Exception as e:
        typer.secho(f"\n✗ Compatibility check failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def summarize(
    outdir: Path = typer.Argument(..., help="Directory with metrics CSVs"),
):
    """
    Summarize CV results and compute averages.

    Example:
        classiflow summarize derived/
    """
    typer.echo(f"Summarizing results in {outdir}...")
    typer.echo("(Placeholder: will compute fold averages and aggregate metrics)")
    # TODO: Implement summarize logic


@app.command()
def export_best(
    outdir: Path = typer.Argument(..., help="Directory with training artifacts"),
):
    """
    Export best-performing task models to spreadsheets.

    Example:
        classiflow export-best derived/
    """
    typer.echo(f"Exporting best models from {outdir}...")
    typer.echo("(Placeholder: will extract best tasks and create summary sheets)")
    # TODO: Implement export logic


@app.command(name="train-hierarchical")
def train_hierarchical_cmd(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset). "
        "Recommended format: .parquet for performance and schema stability.",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features + labels. Use --data instead.",
    ),
    patient_col: Optional[str] = typer.Option(None, "--patient-col", help="Column with patient/slide IDs for stratification (optional; if not provided, sample-level stratification is used)"),
    label_l1: str = typer.Option(..., "--label-l1", help="Level-1 label column"),
    label_l2: Optional[str] = typer.Option(None, "--label-l2", help="Level-2 label column (enables hierarchical mode)"),
    l2_classes: Optional[List[str]] = typer.Option(None, "--l2-classes", help="Subset of L2 classes to include"),
    min_l2_classes_per_branch: int = typer.Option(2, "--min-l2-classes-per-branch", help="Min L2 classes per branch"),
    outdir: Path = typer.Option(Path("derived_hierarchical"), "--outdir", help="Output directory"),
    outer_folds: int = typer.Option(3, "--outer-folds", help="Number of outer CV folds"),
    inner_splits: int = typer.Option(3, "--inner-splits", help="Number of inner CV splits"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
    mlp_epochs: int = typer.Option(100, "--mlp-epochs", help="Max epochs for MLP"),
    mlp_batch_size: int = typer.Option(256, "--mlp-batch-size", help="Batch size"),
    mlp_hidden: int = typer.Option(128, "--mlp-hidden", help="Base hidden dimension"),
    mlp_dropout: float = typer.Option(0.3, "--mlp-dropout", help="Dropout rate"),
    early_stopping_patience: int = typer.Option(10, "--early-stopping-patience", help="Early stopping patience"),
    use_smote: bool = typer.Option(False, "--use-smote", help="Apply SMOTE"),
    smote_k_neighbors: int = typer.Option(5, "--smote-k-neighbors", help="SMOTE k-neighbors"),
    output_format: str = typer.Option("xlsx", "--output-format", help="Output format: xlsx or csv"),
    tracker: Optional[str] = typer.Option(
        None,
        "--tracker",
        help="Experiment tracking backend: mlflow, wandb (requires optional deps)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Experiment/project name for tracking (default: classiflow-hierarchical)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name for tracking (default: auto-generated)",
    ),
    verbose: int = typer.Option(1, "--verbose", help="Verbosity: 0=minimal, 1=standard, 2=detailed"),
):
    """
    Train hierarchical classifier with optional patient-level stratified nested CV.

    Supports:
    - Single-label classification (L1 only)
    - Hierarchical two-level classification (L1 → L2 per branch)
    - Patient-level stratification (no data leakage) when --patient-col is provided
    - Sample-level stratification when --patient-col is not provided
    - PyTorch MLP with CUDA/MPS acceleration
    - Early stopping and hyperparameter tuning
    - CSV, Parquet, and Parquet dataset directories

    Examples:
        # Single parquet file (recommended)
        classiflow train-hierarchical --data data.parquet --label-l1 diagnosis --device auto

        # Dataset directory (chunked parquet)
        classiflow train-hierarchical --data data_parquet/ --patient-col patient_id --label-l1 diagnosis

        # Hierarchical (two-level) with patient stratification
        classiflow train-hierarchical --data data.parquet --patient-col patient_id --label-l1 tumor_type --label-l2 subtype --device cuda

        # Legacy CSV (deprecated)
        classiflow train-hierarchical --data-csv data.csv --label-l1 diagnosis --device auto
    """
    if verbose >= 1:
        logging.getLogger("classiflow").setLevel(logging.DEBUG if verbose >= 2 else logging.INFO)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    config = HierarchicalConfig(
        data_path=resolved_path,
        patient_col=patient_col,
        label_l1=label_l1,
        label_l2=label_l2,
        l2_classes=l2_classes,
        min_l2_classes_per_branch=min_l2_classes_per_branch,
        outdir=outdir,
        outer_folds=outer_folds,
        inner_splits=inner_splits,
        random_state=random_state,
        device=device,
        mlp_epochs=mlp_epochs,
        mlp_batch_size=mlp_batch_size,
        mlp_hidden=mlp_hidden,
        mlp_dropout=mlp_dropout,
        early_stopping_patience=early_stopping_patience,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        output_format=output_format,
        verbose=verbose,
        tracker=tracker,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # Check data compatibility before training
    typer.echo("Checking data compatibility...")
    compat_result = assess_data_compatibility(config, return_details=True)
    print(compat_result)

    if not compat_result.is_compatible:
        typer.secho("\n✗ Data is not compatible with train-hierarchical mode.", fg=typer.colors.RED, err=True)
        typer.echo("\nPlease fix the issues above before training.")
        raise typer.Exit(code=1)

    if compat_result.warnings:
        typer.echo("\n⚠️  Proceeding with warnings (see above)")
        if not typer.confirm("Continue with training?", default=True):
            typer.echo("Training cancelled.")
            raise typer.Exit(code=0)

    try:
        from classiflow.training.hierarchical_cv import train_hierarchical

        results = train_hierarchical(config)
        typer.secho(f"\n✓ Training complete. Results saved to {outdir}", fg=typer.colors.GREEN)
        typer.echo(f"  Folds: {results['n_folds']}")
        typer.echo(f"  L1 classes: {len(results['l1_classes'])}")
        if config.hierarchical:
            typer.echo(f"  L2 branches: {len(results['l2_classes_per_branch'])}")
    except Exception as e:
        typer.secho(f"\n✗ Training failed: {e}", fg=typer.colors.RED, err=True)
        if verbose >= 1:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def infer_hierarchical(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset).",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features. Use --data instead.",
    ),
    model_dir: Path = typer.Option(..., "--model-dir", help="Directory with trained hierarchical models"),
    fold: int = typer.Option(1, "--fold", help="Which fold's model to use (1-indexed)"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
    outfile: Path = typer.Option(Path("predictions.csv"), "--outfile", help="Output predictions CSV"),
    include_proba: bool = typer.Option(True, "--include-proba/--no-proba", help="Include probability columns"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
):
    """
    Run inference with trained hierarchical models.

    Supports both single-label and hierarchical (L1→L2) predictions.
    Supports CSV, Parquet, and Parquet dataset directories.

    Examples:
        # Parquet file (recommended)
        classiflow infer-hierarchical --data test.parquet --model-dir results/ --fold 1

        # With GPU and custom output
        classiflow infer-hierarchical \\
            --data test.parquet \\
            --model-dir results/ \\
            --fold 1 \\
            --device cuda \\
            --outfile predictions.csv

        # Legacy CSV (deprecated)
        classiflow infer-hierarchical --data-csv test.csv --model-dir results/ --fold 1
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        typer.echo(f"Loading models from {model_dir}, fold {fold}...")

        # Load inference object
        from classiflow.inference import HierarchicalInference

        infer = HierarchicalInference(model_dir, fold=fold, device=device)

        typer.echo(f"Model type: {'Hierarchical' if infer.hierarchical else 'Single-label'}")
        typer.echo(f"L1 classes: {', '.join(infer.l1_classes)}")
        if infer.hierarchical:
            typer.echo(f"L2 branches: {len(infer.branch_models)}")

        # Load data using the new unified loader
        from classiflow.data import load_table
        typer.echo(f"\nLoading data from {resolved_path}...")
        df = load_table(resolved_path)

        # Get feature columns (exclude non-numeric)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        typer.echo(f"Using {len(feature_cols)} features")

        X = df[feature_cols].values

        # Predict
        typer.echo(f"\nRunning inference on {len(X)} samples...")
        df_pred = infer.predict_dataframe(X, include_proba=include_proba)

        # Save
        df_pred.to_csv(outfile, index=False)
        typer.secho(f"\n✓ Predictions saved to {outfile}", fg=typer.colors.GREEN)

        # Show summary
        typer.echo(f"\nPrediction summary:")
        if infer.hierarchical:
            typer.echo(df_pred["l1_class"].value_counts().to_string())
        else:
            typer.echo(df_pred["prediction"].value_counts().to_string())

    except Exception as e:
        typer.secho(f"\n✗ Inference failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def infer(
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Path to data file (.csv, .parquet) or directory (parquet dataset).",
    ),
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        help="[DEPRECATED] Path to CSV with features for inference. Use --data instead.",
    ),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", help="Directory containing trained model artifacts"),
    bundle: Optional[Path] = typer.Option(None, "--bundle", help="Bundle ZIP file containing trained model"),
    fold: int = typer.Option(1, "--fold", help="Fold number to use from bundle (default: 1)"),
    outdir: Path = typer.Option(Path("inference_results"), "--outdir", help="Output directory for results"),
    id_col: Optional[str] = typer.Option(None, "--id-col", help="Column name for sample ID (optional)"),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Ground-truth label column for evaluation (optional)"),
    strict: bool = typer.Option(True, "--strict/--lenient", help="Strict mode: fail if features missing; lenient: fill with zeros/median"),
    fill_strategy: str = typer.Option("zero", "--fill-strategy", help="Fill strategy for missing features: zero or median"),
    max_roc_curves: int = typer.Option(10, "--max-roc-curves", help="Max per-class ROC curves to plot"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip generating plots"),
    no_excel: bool = typer.Option(False, "--no-excel", help="Skip generating Excel workbook"),
    device: str = typer.Option("auto", "--device", help="Device for PyTorch models: auto, cpu, cuda, mps"),
    batch_size: int = typer.Option(512, "--batch-size", help="Batch size for inference"),
    verbose: int = typer.Option(1, "--verbose", help="Verbosity: 0=minimal, 1=standard, 2=detailed"),
):
    """
    Run inference on new data using trained models.

    Supports:
    - Binary task models (OvR/pairwise/composite)
    - Meta-classifiers (binary → multiclass)
    - Hierarchical models (L1 → L2 → L3 routing)
    - Automatic feature alignment (strict or lenient mode)
    - Optional evaluation with ground-truth labels
    - Publication-ready outputs (Excel, CSV, plots)
    - Model bundles (portable ZIP archives)
    - CSV, Parquet, and Parquet dataset directories

    Examples:
        # Parquet file (recommended)
        classiflow infer --data test.parquet --run-dir derived/fold1 --outdir results

        # Inference from bundle
        classiflow infer --data test.parquet --bundle models/model.zip --outdir results

        # With evaluation (requires labels)
        classiflow infer --data test.parquet --run-dir derived/fold1 --outdir results --label-col diagnosis

        # Lenient mode (fill missing features)
        classiflow infer --data test.parquet --run-dir derived/fold1 --outdir results --lenient --fill-strategy median

        # Legacy CSV (deprecated)
        classiflow infer --data-csv test.csv --run-dir derived_hierarchical/fold1 --outdir results --device mps
    """
    from classiflow.inference import run_inference, InferenceConfig

    # Set logging level
    log_level = logging.WARNING if verbose == 0 else (logging.INFO if verbose == 1 else logging.DEBUG)
    logging.getLogger("classiflow").setLevel(log_level)

    # Resolve data path (--data takes precedence over --data-csv)
    try:
        resolved_path = _resolve_data_path(data, data_csv)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Validate inputs: must provide either --run-dir or --bundle
    if not run_dir and not bundle:
        typer.secho("Error: Must provide either --run-dir or --bundle", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if run_dir and bundle:
        typer.secho("Error: Cannot provide both --run-dir and --bundle", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Handle bundle extraction
    if bundle:
        from classiflow.bundles import load_bundle

        typer.echo(f"Loading bundle: {bundle}")
        try:
            bundle_data = load_bundle(bundle, fold=fold)
            run_dir = bundle_data["fold_dir"]
            manifest = bundle_data["manifest"]
            typer.echo(f"  Run ID: {manifest.run_id}")
            typer.echo(f"  Task type: {manifest.task_type}")
            typer.echo(f"  Using fold{fold}")
        except Exception as e:
            typer.secho(f"Error: Failed to load bundle: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    try:
        # Create inference config
        config = InferenceConfig(
            run_dir=run_dir,
            data_path=resolved_path,
            output_dir=outdir,
            id_col=id_col,
            label_col=label_col,
            strict_features=strict,
            lenient_fill_strategy=fill_strategy,
            max_roc_curves=max_roc_curves,
            include_plots=not no_plots,
            include_excel=not no_excel,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Run inference
        typer.echo(f"Running inference...")
        typer.echo(f"  Data: {resolved_path}")
        typer.echo(f"  Run directory: {run_dir}")
        typer.echo(f"  Output directory: {outdir}")

        results = run_inference(config)

        # Print summary
        typer.echo("\n" + "="*60)
        typer.secho("✓ Inference completed successfully!", fg=typer.colors.GREEN)
        typer.echo("="*60)
        typer.echo(f"Samples processed: {len(results['predictions'])}")

        typer.echo(f"\nGenerated files:")
        for name, path in results.get("output_files", {}).items():
            typer.echo(f"  • {name}: {path}")

        # Show metrics if computed
        if "metrics" in results and "overall" in results["metrics"]:
            typer.echo(f"\nPerformance Metrics:")
            overall = results["metrics"]["overall"]
            typer.echo(f"  • Accuracy: {overall.get('accuracy', 0):.4f}")
            typer.echo(f"  • Balanced Accuracy: {overall.get('balanced_accuracy', 0):.4f}")
            typer.echo(f"  • F1 (Macro): {overall.get('f1_macro', 0):.4f}")

            if "roc_auc" in overall and "macro" in overall["roc_auc"]:
                typer.echo(f"  • ROC AUC (Macro): {overall['roc_auc']['macro']:.4f}")

        # Show warnings
        if results.get("warnings"):
            typer.echo(f"\n⚠️  Warnings: {len(results['warnings'])}")
            for i, w in enumerate(results["warnings"][:3], 1):
                typer.echo(f"  {i}. {w}")
            if len(results["warnings"]) > 3:
                typer.echo(f"  ... and {len(results['warnings']) - 3} more")

    except Exception as e:
        typer.secho(f"\n✗ Inference failed: {e}", fg=typer.colors.RED, err=True)
        if verbose >= 2:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="compare-smote")
def compare_smote(
    result_dir: Path = typer.Argument(..., help="Directory with training results (contains fold1/, fold2/, etc.)"),
    outdir: Path = typer.Option("smote_analysis", "--outdir", help="Output directory for comparison results"),
    model_type: Optional[str] = typer.Option(None, "--model-type", help="Model type: binary, meta, or hierarchical (auto-detected if None)"),
    metric_file: str = typer.Option("metrics_outer_meta_eval.csv", "--metric-file", help="Name of metrics CSV file to load from each fold"),
    primary_metric: str = typer.Option("f1", "--primary-metric", help="Primary metric for recommendation (f1, accuracy, roc_auc, etc.)"),
    secondary_metric: str = typer.Option("roc_auc", "--secondary-metric", help="Secondary metric for overfitting detection"),
    overfitting_threshold: float = typer.Option(0.03, "--overfitting-threshold", help="Minimum absolute drop to flag overfitting"),
    significance_level: float = typer.Option(0.05, "--significance-level", help="p-value threshold for statistical significance"),
    min_effect_size: float = typer.Option(0.2, "--min-effect-size", help="Minimum Cohen's d for meaningful difference"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """
    Compare SMOTE vs no-SMOTE model performance.

    This command analyzes training results where both SMOTE and no-SMOTE variants
    were trained (using --smote both). It performs:

    - Statistical comparisons (paired t-tests, effect sizes, Wilcoxon tests)
    - Overfitting detection (concurrent performance drops)
    - Publication-ready visualizations (delta charts, scatter plots, distributions)
    - Detailed text and JSON reports with recommendations

    The analysis helps determine whether SMOTE improves model robustness without
    overfitting, providing evidence for publication reviewers.

    Examples:
        # Basic comparison (meta-classifier results)
        classiflow compare-smote derived/results --outdir smote_analysis

        # Specify model type and primary metric
        classiflow compare-smote derived/results --model-type meta --primary-metric f1

        # Custom thresholds for overfitting detection
        classiflow compare-smote derived/results --overfitting-threshold 0.05 --min-effect-size 0.3

        # Hierarchical model comparison
        classiflow compare-smote derived_hierarchical --model-type hierarchical --metric-file metrics.csv

    Output Files:
        - smote_comparison_YYYYMMDD_HHMMSS.txt: Human-readable report
        - smote_comparison_YYYYMMDD_HHMMSS.json: Machine-readable results
        - smote_comparison_summary_YYYYMMDD_HHMMSS.csv: Metric comparisons
        - smote_comparison_delta_bars.png: Performance difference chart
        - smote_comparison_identity_grid.png: Scatter plots for all metrics
        - smote_comparison_dist_*.png: Distribution comparisons per metric
        - smote_comparison_trajectory_*.png: Per-fold performance trajectories
        - smote_comparison_per_task_*.png: Per-task comparisons (if applicable)
    """
    try:
        # Validate model type
        if model_type and model_type not in ["binary", "meta", "hierarchical"]:
            typer.secho(f"Error: Invalid model type '{model_type}'. Must be 'binary', 'meta', or 'hierarchical'",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        # Load comparison
        typer.echo(f"Loading results from: {result_dir}")
        typer.echo(f"  Model type: {model_type or 'auto-detect'}")
        typer.echo(f"  Metric file: {metric_file}")

        comparison = SMOTEComparison.from_directory(
            result_dir,
            model_type=model_type,
            metric_file=metric_file,
        )

        typer.secho(f"✓ Loaded {comparison.n_folds} folds", fg=typer.colors.GREEN)
        typer.echo(f"  Model type: {comparison.model_type}")
        typer.echo(f"  Metrics: {', '.join(comparison.metric_columns)}")

        # Generate report
        typer.echo("\nGenerating comparison report...")
        result = comparison.generate_report(
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            overfitting_threshold=overfitting_threshold,
            significance_level=significance_level,
            min_effect_size=min_effect_size,
        )

        # Print summary to console
        typer.echo("\n" + result.summary_text())

        # Save report files
        typer.echo("\nSaving reports...")
        report_files = comparison.save_report(result, outdir)

        typer.secho(f"\n✓ Reports saved to: {outdir}", fg=typer.colors.GREEN)
        for file_type, file_path in report_files.items():
            typer.echo(f"  • {file_type}: {file_path.name}")

        # Generate plots
        if not no_plots:
            typer.echo("\nGenerating plots...")
            plot_files = comparison.create_all_plots(outdir)

            typer.secho(f"✓ Generated {len(plot_files)} plots", fg=typer.colors.GREEN)
            for plot_type, plot_path in plot_files.items():
                typer.echo(f"  • {plot_type}: {plot_path.name}")

        # Final recommendation summary
        typer.echo("\n" + "="*70)
        typer.secho("RECOMMENDATION SUMMARY", fg=typer.colors.CYAN, bold=True)
        typer.echo("="*70)

        rec_color = {
            "use_smote": typer.colors.GREEN,
            "no_smote": typer.colors.YELLOW,
            "equivalent": typer.colors.BLUE,
            "insufficient_data": typer.colors.RED,
        }

        typer.secho(
            f"\n{result.recommendation.upper().replace('_', ' ')}",
            fg=rec_color.get(result.recommendation, typer.colors.WHITE),
            bold=True
        )
        typer.echo(f"Confidence: {result.confidence.upper()}")

        if result.overfitting_detected:
            typer.secho(
                f"\n⚠️  WARNING: Overfitting detected in {', '.join(result.overfitting_metrics)}",
                fg=typer.colors.RED,
                bold=True
            )
            typer.echo(f"Reason: {result.overfitting_reason}")

        typer.echo("\nKey Findings:")
        for reason in result.reasoning:
            typer.echo(f"  • {reason}")

        typer.echo("\n" + "="*70)
        typer.echo(f"\nFor full details, see: {outdir}/")

    except FileNotFoundError as e:
        typer.secho(f"\n✗ Error: {e}", fg=typer.colors.RED, err=True)
        typer.echo("\nMake sure you trained with --smote both to generate SMOTE/no-SMOTE variants.")
        raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(f"\n✗ Comparison failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
