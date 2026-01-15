"""CLI commands for Classiflow UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

ui_app = typer.Typer(
    name="ui",
    help="Web UI for browsing Classiflow projects and runs.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


@ui_app.command()
def serve(
    projects_root: Path = typer.Option(
        Path("./projects"),
        "--projects-root",
        "-p",
        help="Root directory containing projects",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help="Port to listen on",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        help="SQLite database path (default: .classiflow/ui.db)",
    ),
    static_dir: Optional[Path] = typer.Option(
        None,
        "--static-dir",
        help="Directory containing built frontend",
    ),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Run in development mode (auto-reload, no static serving)",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open browser after starting",
    ),
):
    """
    Start the Classiflow UI server.

    Serves a web interface for browsing projects, runs, and artifacts.

    Examples:
        # Start with default settings
        classiflow ui serve --projects-root ./projects

        # Development mode (backend only, frontend runs separately)
        classiflow ui serve --projects-root ./projects --dev

        # Production with built frontend
        classiflow ui serve --projects-root ./projects --static-dir ./classiflow-ui/dist

        # Open browser automatically
        classiflow ui serve --projects-root ./projects --open
    """
    from classiflow.ui_api.server import run_server, run_dev_server

    typer.echo(f"Starting Classiflow UI...")
    typer.echo(f"  Projects root: {projects_root}")
    typer.echo(f"  Address: http://{host}:{port}")

    if dev:
        typer.echo("  Mode: development (frontend should run separately)")
        run_dev_server(
            projects_root=projects_root,
            host=host,
            port=port,
            db_path=db_path,
        )
    else:
        # Look for built frontend in default locations if not specified
        if not static_dir:
            candidates = [
                Path(__file__).parent.parent.parent.parent / "classiflow-ui" / "dist",
                Path.cwd() / "classiflow-ui" / "dist",
            ]
            for candidate in candidates:
                if candidate.is_dir() and (candidate / "index.html").exists():
                    static_dir = candidate
                    typer.echo(f"  Static files: {static_dir}")
                    break

        if not static_dir:
            typer.echo("  Static files: not found (API only mode)")

        run_server(
            projects_root=projects_root,
            host=host,
            port=port,
            db_path=db_path,
            static_dir=static_dir,
            open_browser=open_browser,
        )


@ui_app.command()
def reindex(
    projects_root: Path = typer.Option(
        Path("./projects"),
        "--projects-root",
        "-p",
        help="Root directory containing projects",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
):
    """
    Rebuild the project index cache.

    Scans the projects directory and rebuilds internal caches.
    Useful after manual file changes or migrations.
    """
    from classiflow.ui_api.scanner import LocalFilesystemScanner

    typer.echo(f"Reindexing projects in {projects_root}...")

    scanner = LocalFilesystemScanner(projects_root)
    projects = scanner.scan_projects(force=True)

    typer.echo(f"\nFound {len(projects)} projects:")
    for project in projects:
        phases = ", ".join(project.phases.keys()) if project.phases else "no runs"
        typer.echo(f"  - {project.id}: {phases} ({project.run_count} runs)")

    typer.secho(f"\n✓ Index complete", fg=typer.colors.GREEN)


@ui_app.command(name="open")
def open_ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8765, "--port"),
):
    """
    Open the Classiflow UI in a web browser.

    Assumes the server is already running.
    """
    import webbrowser

    url = f"http://{host}:{port}"
    typer.echo(f"Opening {url}...")
    webbrowser.open(url)


@ui_app.command()
def init(
    output_dir: Path = typer.Option(
        Path("./classiflow-ui"),
        "--output-dir",
        "-o",
        help="Directory to create frontend scaffold",
    ),
):
    """
    Initialize a new frontend development environment.

    Creates the classiflow-ui directory with React + Tailwind scaffold.
    For development purposes.
    """
    if output_dir.exists():
        typer.secho(f"Directory already exists: {output_dir}", fg=typer.colors.YELLOW)
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    typer.echo(f"Creating frontend scaffold at {output_dir}...")
    typer.echo("Note: Run this command to set up the frontend:")
    typer.echo("")
    typer.echo(f"  cd {output_dir}")
    typer.echo("  npm install")
    typer.echo("  npm run dev")
    typer.echo("")
    typer.secho("See classiflow-ui/README.md for more details", fg=typer.colors.CYAN)


@ui_app.command()
def check(
    projects_root: Path = typer.Option(
        Path("./projects"),
        "--projects-root",
        "-p",
        help="Root directory containing projects",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
):
    """
    Check projects and display diagnostic information.

    Useful for debugging issues with project discovery.
    """
    from classiflow.ui_api.scanner import LocalFilesystemScanner
    from classiflow.ui_api.adapters.manifest import parse_run_manifest

    typer.echo(f"Checking projects in {projects_root}...\n")

    scanner = LocalFilesystemScanner(projects_root)
    projects = scanner.scan_projects(force=True)

    if not projects:
        typer.secho("No projects found!", fg=typer.colors.YELLOW)
        typer.echo("\nExpected structure:")
        typer.echo("  projects/")
        typer.echo("    └── PROJECT_ID__name/")
        typer.echo("        ├── project.yaml")
        typer.echo("        └── runs/")
        typer.echo("            └── technical_validation/")
        typer.echo("                └── run_id/")
        typer.echo("                    └── run.json")
        return

    for project in projects:
        typer.secho(f"\n{'='*60}", fg=typer.colors.CYAN)
        typer.secho(f"Project: {project.id}", fg=typer.colors.GREEN, bold=True)
        typer.secho(f"{'='*60}", fg=typer.colors.CYAN)

        typer.echo(f"  Name: {project.config.name}")
        typer.echo(f"  Task mode: {project.config.task_mode or 'not specified'}")
        typer.echo(f"  Owner: {project.config.owner or 'not specified'}")
        typer.echo(f"  Updated: {project.updated_at or 'unknown'}")

        # Decision
        if project.decision:
            color = typer.colors.GREEN if project.decision.decision == "PASS" else typer.colors.RED
            typer.secho(f"  Decision: {project.decision.decision}", fg=color)
        else:
            typer.echo("  Decision: pending")

        # Phases and runs
        typer.echo(f"\n  Phases ({len(project.phases)}):")
        for phase, run_ids in project.phases.items():
            typer.echo(f"    - {phase}: {len(run_ids)} run(s)")
            if run_ids:
                # Show latest run details
                latest_id = run_ids[0]
                run_dir = projects_root / project.id / "runs" / phase / latest_id
                try:
                    manifest = parse_run_manifest(run_dir, project.id, phase)
                    typer.echo(f"      Latest: {latest_id}")
                    typer.echo(f"        Created: {manifest.created_at}")
                    typer.echo(f"        Task type: {manifest.task_type or 'unknown'}")
                    typer.echo(f"        Source files: {', '.join(manifest.source_files)}")
                except Exception as e:
                    typer.echo(f"      Error reading latest run: {e}")

        # Datasets
        if project.datasets:
            typer.echo(f"\n  Datasets ({len(project.datasets)}):")
            for name, ds in project.datasets.items():
                typer.echo(f"    - {name}: {ds.row_count or '?'} rows, {len(ds.feature_columns)} features")

    typer.echo("")
    typer.secho(f"Total: {len(projects)} projects", fg=typer.colors.GREEN)
