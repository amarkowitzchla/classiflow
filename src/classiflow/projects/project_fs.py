"""Filesystem helpers for project layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def slugify(value: str) -> str:
    """Convert a string to a filesystem-friendly slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "project"


@dataclass
class ProjectPaths:
    """Convenience paths for a classiflow project."""

    root: Path

    @property
    def registry_dir(self) -> Path:
        return self.root / "registry"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def promotion_dir(self) -> Path:
        return self.root / "promotion"

    @property
    def project_yaml(self) -> Path:
        return self.root / "project.yaml"

    @property
    def datasets_yaml(self) -> Path:
        return self.registry_dir / "datasets.yaml"

    @property
    def thresholds_yaml(self) -> Path:
        return self.registry_dir / "thresholds.yaml"

    @property
    def labels_yaml(self) -> Path:
        return self.registry_dir / "labels.yaml"

    @property
    def features_yaml(self) -> Path:
        return self.registry_dir / "features.yaml"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    def ensure(self) -> None:
        """Create base directory layout."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "test").mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.promotion_dir.mkdir(parents=True, exist_ok=True)

    def runs_subdir(self, phase: str, run_id: str) -> Path:
        """Return path for a run phase subdirectory."""
        return self.runs_dir / phase / run_id


def project_root(base_dir: Path, project_id: str, name: str) -> Path:
    """Build the project root path under the base directory."""
    return base_dir / f"{project_id}__{slugify(name)}"


def choose_project_id(name: str, override: Optional[str] = None) -> str:
    """Pick a deterministic-ish project ID when none is provided."""
    if override:
        return override
    base = slugify(name).upper()
    return (base[:12] if base else "PROJECT")
