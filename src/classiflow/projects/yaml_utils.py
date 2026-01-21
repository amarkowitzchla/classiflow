"""YAML helpers for project configuration and registry files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class YamlSupportError(RuntimeError):
    """Raised when YAML support is unavailable."""


def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise YamlSupportError(
            "PyYAML is required for project YAML files. Install with `pip install pyyaml`."
        ) from exc
    return yaml


def load_yaml(path: Path) -> Any:
    """Load YAML from disk."""
    yaml = _require_yaml()
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(data: Any, path: Path) -> None:
    """Write YAML to disk with stable formatting."""
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            default_flow_style=False,
            width=120,
        )
