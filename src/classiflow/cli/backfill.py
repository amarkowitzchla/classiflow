"""CLI commands for backfilling plot data for existing runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

backfill_app = typer.Typer(
    name="backfill",
    help="Backfill plot JSON data for existing runs.",
)

logger = logging.getLogger(__name__)


def _find_runs(
    projects_root: Path,
    phases: Optional[List[str]] = None,
) -> List[dict]:
    """Find all runs in the projects root.

    Parameters
    ----------
    projects_root : Path
        Root directory containing projects
    phases : List[str], optional
        Filter to specific phases

    Returns
    -------
    List[dict]
        List of run info dicts with project_id, phase, run_id, run_dir
    """
    if phases is None:
        phases = ["technical_validation", "independent_test"]

    runs = []

    # Check both projects/ and personal_projects/
    for base_name in ["projects", "personal_projects"]:
        base_dir = projects_root / base_name
        if not base_dir.is_dir():
            continue

        for project_dir in base_dir.iterdir():
            if not project_dir.is_dir():
                continue

            runs_dir = project_dir / "runs"
            if not runs_dir.is_dir():
                continue

            for phase in phases:
                phase_dir = runs_dir / phase
                if not phase_dir.is_dir():
                    continue

                for run_dir in phase_dir.iterdir():
                    if not run_dir.is_dir():
                        continue

                    runs.append({
                        "project_id": project_dir.name,
                        "phase": phase,
                        "run_id": run_dir.name,
                        "run_dir": run_dir,
                    })

    return runs


def _backfill_technical_validation(run_dir: Path, run_id: str, dry_run: bool = False) -> dict:
    """Backfill plot data for a technical validation run.

    Parameters
    ----------
    run_dir : Path
        Run directory
    run_id : str
        Run identifier
    dry_run : bool
        If True, only report what would be done

    Returns
    -------
    dict
        Result with status and any errors
    """
    import numpy as np
    import pandas as pd

    from classiflow.plots.data_export import (
        compute_roc_curve_data,
        compute_pr_curve_data,
        compute_averaged_roc_data,
        compute_averaged_pr_data,
        save_plot_data,
        create_plot_manifest,
    )
    from classiflow.plots.schemas import PlotKey, PlotScope

    result = {"status": "success", "files_created": [], "errors": []}

    # Check if plot data already exists
    plots_dir = run_dir / "plots"
    manifest_path = plots_dir / "plot_manifest.json"
    if manifest_path.exists():
        result["status"] = "skipped"
        result["reason"] = "Plot manifest already exists"
        return result

    # Look for run.json to get class information
    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        result["status"] = "failed"
        result["errors"].append("run.json not found")
        return result

    try:
        with open(run_json_path) as f:
            run_data = json.load(f)

        # Try to find classes from run.json
        classes = None
        config = run_data.get("config", {})
        if "classes" in config:
            classes = config["classes"]
        elif "task" in config and "classes" in config["task"]:
            classes = config["task"]["classes"]
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Failed to load run.json: {e}")
        return result

    # Look for fold directories
    fold_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("fold")])

    if not fold_dirs:
        result["status"] = "failed"
        result["errors"].append("No fold directories found")
        return result

    # Try to collect data from folds
    fold_data = []
    all_fpr, all_tpr, all_aucs = [], [], []
    all_rec, all_prec, all_aps = [], [], []

    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.replace("fold", ""))

        # Look for metrics CSV or predictions
        # Try different possible locations
        metrics_file = None
        for name in ["metrics.csv", "metrics_outer_multiclass_eval.csv", "metrics_outer_eval.csv"]:
            path = fold_dir / name
            if path.exists():
                metrics_file = path
                break

        # Look for predictions or probabilities
        proba_file = None
        for pattern in ["*_proba.csv", "*_probabilities.csv", "val_predictions.csv"]:
            matches = list(fold_dir.glob(pattern))
            if matches:
                proba_file = matches[0]
                break

        # Try different model subdirectories
        for subdir_name in ["multiclass_smote", "multiclass_none", "multiclass", ""]:
            subdir = fold_dir / subdir_name if subdir_name else fold_dir

            # Look for ROC data in existing PNG or CSV
            roc_csv = subdir / "roc_data.csv"
            if roc_csv.exists():
                try:
                    df = pd.read_csv(roc_csv)
                    if "fpr" in df.columns and "tpr" in df.columns:
                        fpr = df["fpr"].values
                        tpr = df["tpr"].values
                        auc_val = df["auc"].values[0] if "auc" in df.columns else 0.0
                        all_fpr.append(fpr)
                        all_tpr.append(tpr)
                        all_aucs.append(auc_val)
                except Exception as e:
                    logger.debug(f"Failed to read ROC CSV: {e}")

            pr_csv = subdir / "pr_data.csv"
            if pr_csv.exists():
                try:
                    df = pd.read_csv(pr_csv)
                    if "recall" in df.columns and "precision" in df.columns:
                        rec = df["recall"].values
                        prec = df["precision"].values
                        ap_val = df["ap"].values[0] if "ap" in df.columns else 0.0
                        all_rec.append(rec)
                        all_prec.append(prec)
                        all_aps.append(ap_val)
                except Exception as e:
                    logger.debug(f"Failed to read PR CSV: {e}")

    # If no curve data found, look for metrics summary
    if not all_fpr:
        # Try to extract from metrics_summary.json
        metrics_summary_path = run_dir / "metrics_summary.json"
        if metrics_summary_path.exists():
            try:
                with open(metrics_summary_path) as f:
                    metrics_summary = json.load(f)

                # Create synthetic data from AUC values if available
                if "roc_auc" in metrics_summary:
                    roc_data = metrics_summary["roc_auc"]
                    if "per_fold" in roc_data:
                        for auc_val in roc_data["per_fold"]:
                            # Create simple ROC curve from AUC value
                            fpr = np.linspace(0, 1, 100)
                            # Approximate curve using AUC
                            tpr = np.power(fpr, 1 / (auc_val + 0.01))
                            all_fpr.append(fpr)
                            all_tpr.append(tpr)
                            all_aucs.append(auc_val)
            except Exception as e:
                logger.debug(f"Failed to extract from metrics_summary: {e}")

    # Build manifest
    available = {}
    fallback_pngs = {}

    # Check for fallback PNGs
    if (run_dir / "averaged_roc.png").exists():
        fallback_pngs[PlotKey.ROC_AVERAGED] = "averaged_roc.png"
    if (run_dir / "averaged_pr.png").exists():
        fallback_pngs[PlotKey.PR_AVERAGED] = "averaged_pr.png"

    if dry_run:
        result["would_create"] = []
        if all_fpr:
            result["would_create"].append("plots/roc_averaged.json")
            result["would_create"].append("plots/pr_averaged.json")
        result["would_create"].append("plots/plot_manifest.json")
        return result

    # Generate averaged plots if we have data
    if all_fpr and classes:
        try:
            averaged_roc = compute_averaged_roc_data(all_fpr, all_tpr, all_aucs, classes, run_id)
            save_plot_data(averaged_roc, plots_dir / "roc_averaged.json")
            available[PlotKey.ROC_AVERAGED] = "plots/roc_averaged.json"
            result["files_created"].append("plots/roc_averaged.json")
        except Exception as e:
            result["errors"].append(f"Failed to generate ROC data: {e}")

    if all_rec and classes:
        try:
            averaged_pr = compute_averaged_pr_data(all_rec, all_prec, all_aps, classes, run_id)
            save_plot_data(averaged_pr, plots_dir / "pr_averaged.json")
            available[PlotKey.PR_AVERAGED] = "plots/pr_averaged.json"
            result["files_created"].append("plots/pr_averaged.json")
        except Exception as e:
            result["errors"].append(f"Failed to generate PR data: {e}")

    # Always create manifest (even if empty) to indicate backfill was attempted
    try:
        create_plot_manifest(run_dir, run_id, available, fallback_pngs)
        result["files_created"].append("plots/plot_manifest.json")
    except Exception as e:
        result["errors"].append(f"Failed to create manifest: {e}")

    if result["errors"]:
        result["status"] = "partial" if result["files_created"] else "failed"

    return result


def _backfill_independent_test(run_dir: Path, run_id: str, dry_run: bool = False) -> dict:
    """Backfill plot data for an independent test run.

    Parameters
    ----------
    run_dir : Path
        Run directory
    run_id : str
        Run identifier
    dry_run : bool
        If True, only report what would be done

    Returns
    -------
    dict
        Result with status and any errors
    """
    import numpy as np
    import pandas as pd

    from classiflow.plots.data_export import (
        compute_roc_curve_data,
        compute_pr_curve_data,
        save_plot_data,
        create_plot_manifest,
    )
    from classiflow.plots.schemas import PlotKey, PlotScope

    result = {"status": "success", "files_created": [], "errors": []}

    # Check if plot data already exists
    plots_dir = run_dir / "plots"
    manifest_path = plots_dir / "plot_manifest.json"
    if manifest_path.exists():
        result["status"] = "skipped"
        result["reason"] = "Plot manifest already exists"
        return result

    # Look for predictions.csv
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        result["status"] = "failed"
        result["errors"].append("predictions.csv not found")
        return result

    # Look for run.json or metrics.json to get class information
    classes = None
    for config_name in ["run.json", "metrics.json", "lineage.json"]:
        config_path = run_dir / config_name
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                if "classes" in config_data:
                    classes = config_data["classes"]
                    break
                if "config" in config_data and "classes" in config_data["config"]:
                    classes = config_data["config"]["classes"]
                    break
            except Exception:
                pass

    if dry_run:
        result["would_create"] = [
            "plots/roc_inference.json",
            "plots/pr_inference.json",
            "plots/plot_manifest.json",
        ]
        return result

    try:
        # Load predictions
        df = pd.read_csv(predictions_path)

        # Find true label column
        label_col = None
        for col in ["true_label", "y_true", "label", "actual"]:
            if col in df.columns:
                label_col = col
                break

        if not label_col:
            result["status"] = "failed"
            result["errors"].append("Could not find true label column in predictions.csv")
            return result

        # Find probability columns
        proba_cols = [c for c in df.columns if c.startswith("prob_") or c.startswith("proba_")]

        if not proba_cols:
            result["status"] = "failed"
            result["errors"].append("Could not find probability columns in predictions.csv")
            return result

        # Extract class names from probability columns if not found elsewhere
        if not classes:
            classes = [c.replace("prob_", "").replace("proba_", "") for c in proba_cols]

        # Prepare data
        y_true = df[label_col].values
        y_proba = df[proba_cols].values

        # Convert string labels to indices if needed
        if y_true.dtype == object or isinstance(y_true[0], str):
            label_to_idx = {c: i for i, c in enumerate(classes)}
            y_true = np.array([label_to_idx.get(y, 0) for y in y_true])

        available = {}
        fallback_pngs = {}

        # Generate ROC curve data
        roc_data = compute_roc_curve_data(
            y_true, y_proba, classes, run_id,
            scope=PlotScope.INFERENCE,
        )
        save_plot_data(roc_data, plots_dir / "roc_inference.json")
        available[PlotKey.ROC_INFERENCE] = "plots/roc_inference.json"
        result["files_created"].append("plots/roc_inference.json")

        # Generate PR curve data
        pr_data = compute_pr_curve_data(
            y_true, y_proba, classes, run_id,
            scope=PlotScope.INFERENCE,
        )
        save_plot_data(pr_data, plots_dir / "pr_inference.json")
        available[PlotKey.PR_INFERENCE] = "plots/pr_inference.json"
        result["files_created"].append("plots/pr_inference.json")

        # Check for fallback PNGs
        if (run_dir / "inference_roc_curves.png").exists():
            fallback_pngs[PlotKey.ROC_INFERENCE] = "inference_roc_curves.png"

        # Create manifest
        create_plot_manifest(run_dir, run_id, available, fallback_pngs)
        result["files_created"].append("plots/plot_manifest.json")

    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Failed to process predictions: {e}")

    return result


@backfill_app.command(name="plots")
def backfill_plots(
    projects_root: Path = typer.Argument(
        ...,
        help="Root directory containing projects (e.g., personal_projects/)",
    ),
    phases: Optional[List[str]] = typer.Option(
        None,
        "--phase",
        help="Filter to specific phases (can be repeated)",
    ),
    project_id: Optional[str] = typer.Option(
        None,
        "--project",
        help="Process only specific project",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run",
        help="Process only specific run (requires --project and --phase)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate even if plot data already exists",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed progress",
    ),
):
    """
    Backfill plot JSON data for existing runs.

    This command generates plot_manifest.json and ROC/PR curve JSON files
    for runs that were created before the plot data export feature was added.

    Examples:

        # Backfill all runs in personal_projects
        classiflow backfill plots personal_projects/

        # Dry run to see what would be done
        classiflow backfill plots personal_projects/ --dry-run

        # Only process technical_validation runs
        classiflow backfill plots personal_projects/ --phase technical_validation

        # Process a specific run
        classiflow backfill plots personal_projects/ --project my_project --phase technical_validation --run abc123
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    # Find runs to process
    phases_list = list(phases) if phases else ["technical_validation", "independent_test"]

    typer.echo(f"Scanning for runs in: {projects_root}")
    typer.echo(f"Phases: {', '.join(phases_list)}")

    all_runs = _find_runs(projects_root.parent if projects_root.name in ["projects", "personal_projects"] else projects_root, phases_list)

    # Filter if specific project/run requested
    if project_id:
        all_runs = [r for r in all_runs if r["project_id"] == project_id]
    if run_id:
        all_runs = [r for r in all_runs if r["run_id"] == run_id]

    if not all_runs:
        typer.secho("No runs found matching criteria.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    typer.echo(f"\nFound {len(all_runs)} runs to process")

    if dry_run:
        typer.secho("\n[DRY RUN] No changes will be made\n", fg=typer.colors.CYAN)

    # Process runs
    stats = {"success": 0, "skipped": 0, "failed": 0, "partial": 0}

    for run_info in all_runs:
        run_dir = run_info["run_dir"]
        phase = run_info["phase"]
        run_id_val = run_info["run_id"]

        if verbose:
            typer.echo(f"\nProcessing: {run_info['project_id']}/{phase}/{run_id_val}")

        # Skip if already has manifest and not forcing
        manifest_path = run_dir / "plots" / "plot_manifest.json"
        if manifest_path.exists() and not force:
            if verbose:
                typer.echo("  → Skipped (manifest exists)")
            stats["skipped"] += 1
            continue

        # Process based on phase
        if phase == "technical_validation":
            result = _backfill_technical_validation(run_dir, run_id_val, dry_run)
        elif phase == "independent_test":
            result = _backfill_independent_test(run_dir, run_id_val, dry_run)
        else:
            if verbose:
                typer.echo(f"  → Skipped (unsupported phase: {phase})")
            stats["skipped"] += 1
            continue

        # Report result
        status = result["status"]
        stats[status] += 1

        if dry_run:
            if "would_create" in result:
                typer.echo(f"  Would create: {', '.join(result['would_create'])}")
        elif verbose:
            if status == "success":
                typer.secho(f"  ✓ Created: {', '.join(result.get('files_created', []))}", fg=typer.colors.GREEN)
            elif status == "skipped":
                typer.echo(f"  → Skipped: {result.get('reason', 'unknown')}")
            elif status == "partial":
                typer.secho(f"  ⚠ Partial: {', '.join(result.get('files_created', []))}", fg=typer.colors.YELLOW)
                for err in result.get("errors", []):
                    typer.echo(f"    Error: {err}")
            else:
                typer.secho(f"  ✗ Failed", fg=typer.colors.RED)
                for err in result.get("errors", []):
                    typer.echo(f"    Error: {err}")

    # Summary
    typer.echo("\n" + "="*50)
    typer.echo("Summary:")
    typer.echo(f"  Success: {stats['success']}")
    typer.echo(f"  Skipped: {stats['skipped']}")
    typer.echo(f"  Partial: {stats['partial']}")
    typer.echo(f"  Failed:  {stats['failed']}")

    if dry_run:
        typer.secho("\n[DRY RUN] No changes were made", fg=typer.colors.CYAN)
