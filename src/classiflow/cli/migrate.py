"""CLI commands for migrating legacy runs to new format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
import typer

logger = logging.getLogger(__name__)

migrate_app = typer.Typer(
    name="migrate",
    help="Migrate legacy runs to new lineage format",
)


@migrate_app.command("run")
def migrate_run(
    run_dir: Path = typer.Argument(..., help="Run directory to migrate"),
    data_csv: Optional[Path] = typer.Option(None, "--data-csv", help="Original training data CSV (for hash computation)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """
    Migrate a legacy run directory to new lineage format.

    This converts run_manifest.json to run.json with full lineage tracking,
    including run_id, data hashes, and feature lists.

    Examples:
        # Dry run (preview changes)
        classiflow migrate run derived --dry-run

        # Migrate with data hash computation
        classiflow migrate run derived --data-csv data.csv

        # Migrate without data file (will generate placeholder hash)
        classiflow migrate run derived
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    try:
        from classiflow.lineage import create_training_manifest, TrainingRunManifest
        from classiflow.lineage.hashing import get_file_metadata
        import uuid

        typer.echo(f"Migrating run directory: {run_dir}")

        # Check for existing files
        legacy_manifest = run_dir / "run_manifest.json"
        new_manifest = run_dir / "run.json"

        if not legacy_manifest.exists():
            typer.secho(f"✗ No run_manifest.json found in {run_dir}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        if new_manifest.exists() and not dry_run:
            typer.secho(f"⚠ run.json already exists, will overwrite", fg=typer.colors.YELLOW)

        # Load legacy manifest
        with open(legacy_manifest, "r") as f:
            legacy_data = json.load(f)

        typer.echo(f"\nLegacy manifest loaded:")
        typer.echo(f"  Config: {legacy_data.get('config', {}).keys()}")

        # Compute data hash if CSV provided
        data_hash = None
        data_size = 0
        data_rows = None

        if data_csv and data_csv.exists():
            typer.echo(f"\nComputing data hash from: {data_csv}")
            metadata = get_file_metadata(data_csv)
            data_hash = metadata["sha256_hash"]
            data_size = metadata["size_bytes"]
            data_rows = metadata.get("row_count")
            typer.echo(f"  Hash: {data_hash[:16]}...")
            typer.echo(f"  Size: {data_size / 1024:.1f} KB")
            typer.echo(f"  Rows: {data_rows}")
        else:
            typer.secho("⚠ No data CSV provided, using placeholder hash", fg=typer.colors.YELLOW)
            data_hash = "0" * 64
            data_size = 0
            data_rows = None

        # Detect task type
        config = legacy_data.get("config", {})
        if "label_l1" in config:
            task_type = "hierarchical"
        elif "classes" in config or "tasks_json" in config:
            task_type = "meta"
        else:
            task_type = "binary"

        # Extract feature list (if available)
        feature_list = []
        # Try to find feature list in various places
        if "feature_list" in legacy_data:
            feature_list = legacy_data["feature_list"] or []
        elif "feature_cols" in config:
            feature_list = config["feature_cols"] or []

        typer.echo(f"\nDetected:")
        typer.echo(f"  Task type: {task_type}")
        typer.echo(f"  Features: {len(feature_list) if feature_list else 0}")

        # Create new manifest
        data_path = Path(config.get("data_csv", "unknown"))

        new_manifest_obj = create_training_manifest(
            data_path=data_path,
            data_hash=data_hash,
            data_size_bytes=data_size,
            data_row_count=data_rows,
            config=config,
            task_type=task_type,
            feature_list=feature_list,
        )

        # Preserve original run_id if it exists
        if "run_id" in legacy_data:
            new_manifest_obj.run_id = legacy_data["run_id"]
            typer.echo(f"  Preserved existing run_id: {new_manifest_obj.run_id}")
        else:
            typer.echo(f"  Generated new run_id: {new_manifest_obj.run_id}")

        if dry_run:
            typer.echo("\n" + "="*60)
            typer.secho("DRY RUN - No changes made", fg=typer.colors.YELLOW)
            typer.echo("="*60)
            typer.echo("\nWould create run.json with:")
            typer.echo(json.dumps(new_manifest_obj.to_dict(), indent=2))
        else:
            # Save new manifest
            new_manifest_obj.save(new_manifest)
            typer.secho(f"\n✓ Migration complete!", fg=typer.colors.GREEN)
            typer.echo(f"  Created: {new_manifest}")
            typer.echo(f"  Run ID: {new_manifest_obj.run_id}")

            # Optionally backup legacy manifest
            backup_path = run_dir / "run_manifest.json.backup"
            if not backup_path.exists():
                import shutil
                shutil.copy(legacy_manifest, backup_path)
                typer.echo(f"  Backup: {backup_path}")

    except Exception as e:
        typer.secho(f"\n✗ Migration failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@migrate_app.command("batch")
def migrate_batch(
    parent_dir: Path = typer.Argument(..., help="Parent directory containing multiple run directories"),
    pattern: str = typer.Option("*", "--pattern", help="Glob pattern for run directories"),
    data_csv: Optional[Path] = typer.Option(None, "--data-csv", help="Training data CSV"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """
    Batch migrate multiple run directories.

    Examples:
        # Migrate all derived_* directories
        classiflow migrate batch . --pattern "derived_*" --dry-run
    """
    if verbose:
        logging.getLogger("classiflow").setLevel(logging.DEBUG)

    run_dirs = list(parent_dir.glob(pattern))
    run_dirs = [d for d in run_dirs if d.is_dir() and (d / "run_manifest.json").exists()]

    if not run_dirs:
        typer.secho(f"✗ No run directories found matching pattern: {pattern}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(run_dirs)} run directories to migrate:")
    for rd in run_dirs:
        typer.echo(f"  - {rd.name}")

    if not dry_run:
        confirm = typer.confirm(f"\nProceed with migration?")
        if not confirm:
            typer.echo("Cancelled")
            raise typer.Exit(code=0)

    success_count = 0
    fail_count = 0

    for run_dir in run_dirs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Migrating: {run_dir.name}")
        typer.echo(f"{'='*60}")

        try:
            # Call migrate_run for each directory
            from typer.testing import CliRunner
            runner = CliRunner()

            cmd = ["run", str(run_dir)]
            if data_csv:
                cmd.extend(["--data-csv", str(data_csv)])
            if dry_run:
                cmd.append("--dry-run")

            result = runner.invoke(migrate_app, cmd)

            if result.exit_code == 0:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            typer.secho(f"✗ Failed: {e}", fg=typer.colors.RED)
            fail_count += 1

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Batch migration complete:")
    typer.secho(f"  Success: {success_count}", fg=typer.colors.GREEN)
    if fail_count > 0:
        typer.secho(f"  Failed: {fail_count}", fg=typer.colors.RED)
    typer.echo(f"{'='*60}")
