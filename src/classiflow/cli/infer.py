"""CLI command for running inference."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional
import click

from classiflow.inference import run_inference, InferenceConfig

logger = logging.getLogger(__name__)


@click.command("infer")
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing trained model artifacts",
)
@click.option(
    "--bundle",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Bundle ZIP file containing trained model",
)
@click.option(
    "--fold",
    type=int,
    default=1,
    help="Fold number to use from bundle (default: 1)",
)
@click.option(
    "--data-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input CSV file with features for inference",
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for inference results",
)
@click.option(
    "--id-col",
    default=None,
    help="Column name for sample ID (optional)",
)
@click.option(
    "--label-col",
    default=None,
    help="Ground-truth label column for evaluation (optional)",
)
@click.option(
    "--strict/--lenient",
    "strict_features",
    default=True,
    help="Strict mode: fail if required features missing. Lenient mode: fill with zeros/median.",
)
@click.option(
    "--fill-strategy",
    type=click.Choice(["zero", "median"]),
    default="zero",
    help="Strategy for filling missing features in lenient mode (default: zero)",
)
@click.option(
    "--max-roc-curves",
    type=int,
    default=10,
    help="Maximum number of per-class ROC curves to generate (default: 10)",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Skip generating plots",
)
@click.option(
    "--no-excel",
    is_flag=True,
    help="Skip generating Excel workbook",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device for PyTorch models (default: auto)",
)
@click.option(
    "--batch-size",
    type=int,
    default=512,
    help="Batch size for inference (default: 512)",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (-v, -vv, -vvv)",
)
def infer_command(
    run_dir: Optional[Path],
    bundle: Optional[Path],
    fold: int,
    data_csv: Path,
    outdir: Path,
    id_col: str | None,
    label_col: str | None,
    strict_features: bool,
    fill_strategy: str,
    max_roc_curves: int,
    no_plots: bool,
    no_excel: bool,
    device: str,
    batch_size: int,
    verbose: int,
):
    """
    Run inference on new data using trained models.

    Examples:

    \b
    # Basic inference from run directory (no evaluation)
    classiflow infer --run-dir derived/fold1 --data-csv test.csv --outdir results

    \b
    # Inference from bundle
    classiflow infer --bundle models/model.zip --data-csv test.csv --outdir results

    \b
    # With evaluation (requires ground-truth labels)
    classiflow infer --run-dir derived/fold1 --data-csv test.csv --outdir results --label-col diagnosis

    \b
    # Lenient mode (fill missing features)
    classiflow infer --run-dir derived/fold1 --data-csv test.csv --outdir results --lenient --fill-strategy median

    \b
    # With sample IDs
    classiflow infer --run-dir derived/fold1 --data-csv test.csv --outdir results --id-col patient_id

    """
    # Configure logging
    log_level = logging.WARNING
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate inputs: must provide either --run-dir or --bundle
    if not run_dir and not bundle:
        click.echo("Error: Must provide either --run-dir or --bundle", err=True)
        sys.exit(1)

    if run_dir and bundle:
        click.echo("Error: Cannot provide both --run-dir and --bundle", err=True)
        sys.exit(1)

    # Handle bundle extraction
    bundle_loader = None
    if bundle:
        from classiflow.bundles import load_bundle

        click.echo(f"Loading bundle: {bundle}")
        try:
            bundle_data = load_bundle(bundle, fold=fold)
            run_dir = bundle_data["fold_dir"]
            manifest = bundle_data["manifest"]
            click.echo(f"  Run ID: {manifest.run_id}")
            click.echo(f"  Task type: {manifest.task_type}")
            click.echo(f"  Using fold{fold}")
            logger.info(f"Loaded bundle: {bundle} (fold{fold})")
        except Exception as e:
            click.echo(f"Error: Failed to load bundle: {e}", err=True)
            sys.exit(1)

    # Create config
    try:
        config = InferenceConfig(
            run_dir=run_dir,
            data_csv=data_csv,
            output_dir=outdir,
            id_col=id_col,
            label_col=label_col,
            strict_features=strict_features,
            lenient_fill_strategy=fill_strategy,
            max_roc_curves=max_roc_curves,
            include_plots=not no_plots,
            include_excel=not no_excel,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
        )
    except Exception as e:
        click.echo(f"Error: Failed to create configuration: {e}", err=True)
        sys.exit(1)

    # Run inference
    try:
        results = run_inference(config)

        # Print summary
        click.echo("\n" + "="*60)
        click.echo("Inference completed successfully!")
        click.echo("="*60)
        click.echo(f"Samples processed: {len(results['predictions'])}")
        click.echo(f"Output directory: {config.output_dir}")
        click.echo(f"\nGenerated files:")

        for name, path in results.get("output_files", {}).items():
            click.echo(f"  - {name}: {path}")

        # Show metrics if computed
        if "metrics" in results and "overall" in results["metrics"]:
            click.echo(f"\nMetrics:")
            overall = results["metrics"]["overall"]
            click.echo(f"  - Accuracy: {overall.get('accuracy', 'N/A'):.4f}")
            click.echo(f"  - Balanced Accuracy: {overall.get('balanced_accuracy', 'N/A'):.4f}")
            click.echo(f"  - F1 (Macro): {overall.get('f1_macro', 'N/A'):.4f}")

            if "roc_auc" in overall:
                roc = overall["roc_auc"]
                if "macro" in roc:
                    click.echo(f"  - ROC AUC (Macro): {roc['macro']:.4f}")

        # Show warnings
        if results.get("warnings"):
            click.echo(f"\nWarnings: {len(results['warnings'])}")
            for i, w in enumerate(results["warnings"][:3], 1):
                click.echo(f"  {i}. {w}")
            if len(results["warnings"]) > 3:
                click.echo(f"  ... and {len(results['warnings']) - 3} more")

        sys.exit(0)

    except Exception as e:
        click.echo(f"\nError: Inference failed: {e}", err=True)
        if verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)
