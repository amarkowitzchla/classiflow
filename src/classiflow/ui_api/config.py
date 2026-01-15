"""Configuration for UI API server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class StorageMode(str, Enum):
    """Storage backend mode."""

    LOCAL = "local"
    POSTGRES = "postgres"
    DATABRICKS = "databricks"


@dataclass
class UIConfig:
    """Configuration for the UI API server."""

    # Storage
    projects_root: Path = field(default_factory=lambda: Path("./projects"))
    storage_mode: StorageMode = StorageMode.LOCAL
    db_path: Path = field(default_factory=lambda: Path(".classiflow/ui.db"))

    # Server
    host: str = "127.0.0.1"
    port: int = 8765
    reload: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    # Optional Postgres (for comments/reviews)
    postgres_url: Optional[str] = None

    # Optional Databricks (read-only metadata)
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None
    databricks_catalog: Optional[str] = None
    databricks_schema: Optional[str] = None

    @classmethod
    def from_env(cls) -> UIConfig:
        """Create configuration from environment variables."""
        config = cls()

        # Projects root
        if projects_root := os.environ.get("CLASSIFLOW_PROJECTS_ROOT"):
            config.projects_root = Path(projects_root)

        # Storage mode
        if storage_mode := os.environ.get("CLASSIFLOW_STORAGE_MODE"):
            try:
                config.storage_mode = StorageMode(storage_mode.lower())
            except ValueError:
                pass

        # Database path
        if db_path := os.environ.get("CLASSIFLOW_DB_PATH"):
            config.db_path = Path(db_path)

        # Server settings
        if host := os.environ.get("CLASSIFLOW_UI_HOST"):
            config.host = host
        if port := os.environ.get("CLASSIFLOW_UI_PORT"):
            try:
                config.port = int(port)
            except ValueError:
                pass

        # CORS
        if cors_origins := os.environ.get("CLASSIFLOW_CORS_ORIGINS"):
            config.cors_origins = [o.strip() for o in cors_origins.split(",")]

        # Postgres
        config.postgres_url = os.environ.get("CLASSIFLOW_POSTGRES_URL")

        # Databricks
        config.databricks_host = os.environ.get("DATABRICKS_HOST")
        config.databricks_token = os.environ.get("DATABRICKS_TOKEN")
        config.databricks_catalog = os.environ.get("CLASSIFLOW_DATABRICKS_CATALOG")
        config.databricks_schema = os.environ.get("CLASSIFLOW_DATABRICKS_SCHEMA")

        return config

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.projects_root.is_dir():
            errors.append(f"Projects root does not exist: {self.projects_root}")

        if self.storage_mode == StorageMode.POSTGRES and not self.postgres_url:
            errors.append("Postgres mode requires CLASSIFLOW_POSTGRES_URL")

        if self.storage_mode == StorageMode.DATABRICKS:
            if not self.databricks_host:
                errors.append("Databricks mode requires DATABRICKS_HOST")
            if not self.databricks_token:
                errors.append("Databricks mode requires DATABRICKS_TOKEN")

        return errors
