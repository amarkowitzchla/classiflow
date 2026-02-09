"""Configuration helper commands for project.yaml discoverability and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
import yaml

from classiflow.projects.project_models import (
    ProjectConfig,
    available_project_options,
    normalize_project_payload,
    validate_project_payload,
)
from classiflow.projects.yaml_utils import load_yaml

config_app = typer.Typer(
    name="config",
    help="Inspect, explain, validate, and normalize Classiflow project configs.",
    add_completion=False,
)


def _resolve_ref(schema: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    while "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/$defs/"):
            break
        key = ref.split("/")[-1]
        defs = schema.get("$defs", {})
        resolved = defs.get(key)
        if not isinstance(resolved, dict):
            break
        node = resolved
    return node


def _schema_node_for_path(schema: Dict[str, Any], field_path: str) -> Optional[Dict[str, Any]]:
    node: Dict[str, Any] = schema
    for part in field_path.split("."):
        node = _resolve_ref(schema, node)
        props = node.get("properties", {})
        if part not in props:
            return None
        child = props[part]
        if not isinstance(child, dict):
            return None
        node = child
    return _resolve_ref(schema, node)


@config_app.command("show")
def config_show(
    mode: str = typer.Option(..., "--mode", help="Task mode"),
    engine: str = typer.Option(..., "--engine", help="Execution engine"),
    device: Optional[str] = typer.Option(None, "--device", help="Device for torch/hybrid engine"),
):
    """Print a mode/engine-aware default project.yaml template."""
    config = ProjectConfig.scaffold(
        project_id="PROJECT_ID",
        name="Project Name",
        mode=mode.lower(),  # type: ignore[arg-type]
        engine=engine.lower(),  # type: ignore[arg-type]
        device=device.lower() if device else None,  # type: ignore[arg-type]
        train_manifest="data/train/manifest.csv",
        test_manifest="data/test/manifest.csv",
    )
    typer.echo(yaml.safe_dump(config.to_yaml_dict(minimal=True), sort_keys=False))


@config_app.command("explain")
def config_explain(field_path: str = typer.Argument(..., help="Dot path, e.g. execution.engine")):
    """Explain a config field: description, allowed values, and default."""
    schema = ProjectConfig.model_json_schema()
    node = _schema_node_for_path(schema, field_path)
    if node is None:
        typer.secho(f"Unknown field path: {field_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    description = node.get("description") or "(no description available)"
    typer.echo(f"field: {field_path}")
    typer.echo(f"description: {description}")

    enum_values = node.get("enum")
    if isinstance(enum_values, list) and enum_values:
        typer.echo(f"allowed: {', '.join(str(v) for v in enum_values)}")

    if "default" in node:
        typer.echo(f"default: {node['default']}")

    options = available_project_options()
    if field_path in options:
        typer.echo(f"common values: {', '.join(options[field_path])}")


@config_app.command("validate")
def config_validate(config_path: Path = typer.Argument(..., help="Path to project.yaml")):
    """Validate project.yaml and emit actionable errors."""
    payload = load_yaml(config_path)
    _, migration_warnings, errors = validate_project_payload(payload)

    for warning_msg in migration_warnings:
        typer.secho(f"warning: {warning_msg}", fg=typer.colors.YELLOW)

    if errors:
        typer.secho("validation failed:", fg=typer.colors.RED, err=True)
        for err in errors:
            typer.secho(f"  - {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("validation passed", fg=typer.colors.GREEN)


@config_app.command("normalize")
def config_normalize(
    config_path: Path = typer.Argument(..., help="Input project.yaml"),
    out: Path = typer.Option(..., "--out", help="Output normalized YAML path"),
):
    """Normalize legacy config keys into the new execution-aware schema."""
    payload = load_yaml(config_path)
    normalized, migration_warnings = normalize_project_payload(payload)

    config = ProjectConfig.model_validate(normalized)
    config.save(out, minimal=True)

    for warning_msg in migration_warnings:
        typer.secho(f"warning: {warning_msg}", fg=typer.colors.YELLOW)

    typer.secho(f"normalized config written to {out}", fg=typer.colors.GREEN)
