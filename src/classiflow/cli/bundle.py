"""CLI commands for model bundle management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import typer

from classiflow.bundles import create_bundle, inspect_bundle, print_bundle_info
from classiflow import __version__

logger = logging.getLogger(__name__)

bundle_app = typer.Typer(
    name="bundle",
    help="Model bundle management (create, inspect, validate)",
)


@bundle_app.command("create")
def create_bundle_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", help="Training run directory"),
    out: Path = typer.Option(..., "--out", help="Output bundle path (will add .zip if missing)"),
    fold: Optional[int] = typer.Option(None, "--fold", help="Specific fold to include (default: fold 1)"),
    all_folds: bool = typer.Option(False, "--all-folds", help="Include all folds"),
    include_metrics: bool = typer.Option(True, "--include-metrics/--no-metrics", help="Include metrics CSVs"),
    description: Optional[str] = typer.Option(None, "--description", help="Bundle description"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """
    Create a portable model bundle from a training run.

    Bundles are self-contained ZIP archives that include:
    - run.json (training manifest with run_id, data hash, etc.)
    - artifacts.json (artifact registry)
    - version.txt (package version)
    - README.txt (usage instructions)
    - fold directories with serialized models
    - optional: metrics CSVs

    Examples:
        # Bundle a single fold
        classiflow bundle create --run-dir derived/fold1 --out my_model.zip

        # Bundle all folds
        classiflow bundle create --run-dir derived --out my_model.zip --all-folds

        # Add description
        classiflow bundle create --run-dir derived --out my_model.zip --description "Production model v1.2"
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    try:
        typer.echo(f"Creating bundle from {run_dir}...")

        bundle_path = create_bundle(
            run_dir=run_dir,
            out_bundle=out,
            fold=fold,
            include_all_folds=all_folds,
            include_metrics=include_metrics,
            description=description,
        )

        typer.secho(f"\n✓ Bundle created: {bundle_path}", fg=typer.colors.GREEN)

        # Show size
        size_mb = bundle_path.stat().st_size / (1024 * 1024)
        typer.echo(f"  Size: {size_mb:.2f} MB")

    except Exception as e:
        typer.secho(f"\n✗ Bundle creation failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@bundle_app.command("inspect")
def inspect_bundle_cmd(
    bundle: Path = typer.Argument(..., help="Path to bundle ZIP file"),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed file listing"),
):
    """
    Inspect a model bundle and display metadata.

    Shows:
    - Run ID and training metadata
    - Included folds and artifacts
    - Package version
    - Bundle size and file count

    Examples:
        # Basic inspection
        classiflow bundle inspect my_model.zip

        # Verbose (show all files)
        classiflow bundle inspect my_model.zip --verbose
    """
    try:
        print_bundle_info(bundle, verbose=verbose)

    except Exception as e:
        typer.secho(f"\n✗ Inspection failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@bundle_app.command("validate")
def validate_bundle_cmd(
    bundle: Path = typer.Argument(..., help="Path to bundle ZIP file"),
):
    """
    Validate a bundle for compatibility and completeness.

    Checks:
    - Required files present (run.json, artifacts.json, version.txt)
    - Version compatibility
    - Manifest integrity

    Examples:
        classiflow bundle validate my_model.zip
    """
    try:
        from classiflow.bundles.inspect import validate_bundle_version
        import classiflow

        typer.echo(f"Validating bundle: {bundle}")

        # Inspect
        info = inspect_bundle(bundle)

        typer.echo(f"\nBundle: {info['bundle_path']}")
        typer.echo(f"Size: {info['size_mb']:.2f} MB")
        typer.echo(f"Files: {info['file_count']}")

        # Validate
        if not info.get("valid"):
            typer.secho("\n✗ Invalid bundle", fg=typer.colors.RED)
            typer.echo(f"Missing files: {', '.join(info.get('missing_files', []))}")
            raise typer.Exit(code=1)

        # Version compatibility
        compatible, warnings = validate_bundle_version(bundle, __version__)

        if compatible:
            typer.secho("\n✓ Bundle is valid and compatible", fg=typer.colors.GREEN)
        else:
            typer.secho("\n⚠ Bundle has compatibility warnings", fg=typer.colors.YELLOW)

        if warnings:
            typer.echo("\nWarnings:")
            for w in warnings:
                typer.echo(f"  - {w}")

    except Exception as e:
        typer.secho(f"\n✗ Validation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
