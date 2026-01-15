"""Server runner for Classiflow UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import uvicorn

from classiflow.ui_api.app import create_app, mount_static
from classiflow.ui_api.config import UIConfig

logger = logging.getLogger(__name__)


def run_server(
    projects_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: Optional[Path] = None,
    static_dir: Optional[Path] = None,
    reload: bool = False,
    open_browser: bool = False,
):
    """
    Run the UI server.

    Parameters
    ----------
    projects_root : Path
        Root directory containing projects
    host : str
        Host to bind to
    port : int
        Port to listen on
    db_path : Path, optional
        SQLite database path for comments/reviews
    static_dir : Path, optional
        Directory containing built frontend
    reload : bool
        Enable auto-reload for development
    open_browser : bool
        Open browser after starting
    """
    # Build config
    config = UIConfig(
        projects_root=Path(projects_root).resolve(),
        host=host,
        port=port,
        reload=reload,
    )

    if db_path:
        config.db_path = Path(db_path)

    # Validate
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Configuration error: {err}")
        raise SystemExit(1)

    # Create app
    app = create_app(config)

    # Mount static files if provided
    if static_dir:
        mount_static(app, Path(static_dir))

    # Open browser
    if open_browser:
        import webbrowser
        import threading

        def _open():
            import time
            time.sleep(1)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    # Run server
    logger.info(f"Starting Classiflow UI at http://{host}:{port}")
    logger.info(f"Projects root: {config.projects_root}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_dev_server(
    projects_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: Optional[Path] = None,
):
    """
    Run the UI server in development mode.

    This mode:
    - Enables auto-reload
    - Doesn't serve static files (frontend runs separately with Vite)
    - Uses broader CORS for local development
    """
    config = UIConfig(
        projects_root=Path(projects_root).resolve(),
        host=host,
        port=port,
        reload=True,
        cors_origins=["*"],  # Allow all origins in dev mode
    )

    if db_path:
        config.db_path = Path(db_path)

    # Validate
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Configuration error: {err}")
        raise SystemExit(1)

    logger.info(f"Starting Classiflow UI (dev mode) at http://{host}:{port}")
    logger.info(f"Projects root: {config.projects_root}")
    logger.info("Frontend should run separately: cd classiflow-ui && npm run dev")

    # Use string import path for reload
    uvicorn.run(
        "classiflow.ui_api.app:create_app",
        host=host,
        port=port,
        reload=True,
        factory=True,
        log_level="debug",
    )
