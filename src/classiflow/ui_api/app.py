"""FastAPI application for Classiflow UI."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

import classiflow
from classiflow.ui_api.config import UIConfig, StorageMode
from classiflow.ui_api.models import (
    Artifact,
    Comment,
    CommentCreate,
    HealthResponse,
    PaginatedResponse,
    ProjectCard,
    ProjectDashboard,
    Review,
    ReviewCreate,
    ReviewStatus,
    RunBrief,
    RunDetail,
)
from classiflow.ui_api.repositories.interfaces import (
    ArtifactRepository,
    CommentRepository,
    ProjectRepository,
    ReviewRepository,
    RunRepository,
)
from classiflow.ui_api.repositories.local import LocalFilesystemRepository
from classiflow.ui_api.repositories.sqlite import SQLiteCommentReviewRepository

logger = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """Static files with SPA fallback for client-side routes."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code != 404:
            return response

        request = Request(scope)
        accept = request.headers.get("accept", "")
        if "." in path and "text/html" not in accept:
            return response

        return await super().get_response("index.html", scope)


def create_app(config: Optional[UIConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    config : UIConfig, optional
        Server configuration. If not provided, loads from environment.

    Returns
    -------
    FastAPI
        Configured application instance
    """
    if config is None:
        config = UIConfig.from_env()

    # Validate config
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    app = FastAPI(
        title="Classiflow UI API",
        description="API for browsing ML projects, runs, and artifacts",
        version=classiflow.__version__,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize repositories based on storage mode
    if config.storage_mode == StorageMode.LOCAL:
        # Local filesystem for projects/runs/artifacts
        local_repo = LocalFilesystemRepository(config.projects_root)
        project_repo: ProjectRepository = local_repo
        run_repo: RunRepository = local_repo
        artifact_repo: ArtifactRepository = local_repo

        # SQLite for comments/reviews
        comment_review_repo = SQLiteCommentReviewRepository(config.db_path)
        comment_repo: CommentRepository = comment_review_repo
        review_repo: ReviewRepository = comment_review_repo

    else:
        # Postgres/Databricks modes would go here
        raise NotImplementedError(f"Storage mode {config.storage_mode} not yet implemented")

    # Store config and repos in app state
    app.state.config = config
    app.state.project_repo = project_repo
    app.state.run_repo = run_repo
    app.state.artifact_repo = artifact_repo
    app.state.comment_repo = comment_repo
    app.state.review_repo = review_repo

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI):
    """Register all API routes."""

    # -------------------------------------------------------------------------
    # Health endpoint
    # -------------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Check service health and configuration."""
        config: UIConfig = app.state.config
        project_repo: ProjectRepository = app.state.project_repo

        # Get project count
        projects, total = project_repo.list_projects(page_size=1)

        return HealthResponse(
            status="ok",
            storage_mode=config.storage_mode.value,
            projects_root=str(config.projects_root.resolve()),
            db_path=str(config.db_path) if config.storage_mode == StorageMode.LOCAL else None,
            project_count=total,
            index_status="ready",
            version=classiflow.__version__,
        )

    # -------------------------------------------------------------------------
    # Project endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/projects", response_model=PaginatedResponse, tags=["Projects"])
    async def list_projects(
        q: Optional[str] = Query(None, description="Search query"),
        mode: Optional[str] = Query(None, description="Task mode filter: binary, meta, hierarchical"),
        owner: Optional[str] = Query(None, description="Owner filter"),
        updated_after: Optional[str] = Query(None, description="ISO timestamp filter"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    ):
        """List all projects with optional filtering."""
        project_repo: ProjectRepository = app.state.project_repo
        projects, total = project_repo.list_projects(
            query=q,
            mode=mode,
            owner=owner,
            updated_after=updated_after,
            page=page,
            page_size=page_size,
        )

        return PaginatedResponse(
            items=[p.model_dump() for p in projects],
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total,
            has_prev=page > 1,
        )

    @app.get("/api/projects/{project_id}", response_model=ProjectDashboard, tags=["Projects"])
    async def get_project(project_id: str):
        """Get project dashboard with full details."""
        project_repo: ProjectRepository = app.state.project_repo
        project = project_repo.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        return project

    @app.get("/api/projects/{project_id}/runs", tags=["Projects"])
    async def get_project_runs(project_id: str):
        """Get runs grouped by phase for a project."""
        project_repo: ProjectRepository = app.state.project_repo
        runs = project_repo.get_project_runs(project_id)
        if not runs:
            # Check if project exists
            project = project_repo.get_project(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        return runs

    # -------------------------------------------------------------------------
    # Run endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/runs/{run_key}", response_model=RunDetail, tags=["Runs"])
    async def get_run(run_key: str):
        """Get run detail by composite key (project:phase:run_id)."""
        run_repo: RunRepository = app.state.run_repo
        run = run_repo.get_run(run_key)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_key}")
        return run

    @app.get("/api/runs/{run_key}/artifacts", response_model=PaginatedResponse, tags=["Runs"])
    async def list_run_artifacts(
        run_key: str,
        kind: Optional[str] = Query(None, description="Filter by artifact kind"),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
    ):
        """List artifacts for a run."""
        artifact_repo: ArtifactRepository = app.state.artifact_repo
        artifacts, total = artifact_repo.list_artifacts(
            run_key=run_key,
            kind=kind,
            page=page,
            page_size=page_size,
        )

        return PaginatedResponse(
            items=[a.model_dump() for a in artifacts],
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total,
            has_prev=page > 1,
        )

    # -------------------------------------------------------------------------
    # Artifact endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/artifacts/{artifact_id}", response_model=Artifact, tags=["Artifacts"])
    async def get_artifact(artifact_id: str):
        """Get artifact metadata."""
        artifact_repo: ArtifactRepository = app.state.artifact_repo
        artifact = artifact_repo.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
        return artifact

    @app.get("/api/artifacts/{artifact_id}/content", tags=["Artifacts"])
    async def get_artifact_content(
        artifact_id: str,
        download: bool = Query(False, description="Force download instead of inline view"),
    ):
        """
        Get artifact content for viewing or download.

        Security: Only serves files with allowed extensions under projects_root.
        """
        artifact_repo: ArtifactRepository = app.state.artifact_repo

        # Get artifact metadata first
        artifact = artifact_repo.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        # Get safe path
        path = artifact_repo.get_artifact_path(artifact_id)
        if not path or not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found")

        # Determine content type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        # Determine disposition
        if download:
            media_type = "application/octet-stream"
            disposition = f'attachment; filename="{path.name}"'
        else:
            media_type = mime_type
            disposition = f'inline; filename="{path.name}"'

        return FileResponse(
            path=path,
            media_type=media_type,
            headers={"Content-Disposition": disposition},
        )

    # Alternative path-based artifact access (more discoverable URLs)
    @app.get(
        "/api/projects/{project_id}/runs/{phase}/{run_id}/artifacts/{artifact_path:path}",
        tags=["Artifacts"],
    )
    async def get_artifact_by_path(
        project_id: str,
        phase: str,
        run_id: str,
        artifact_path: str,
        download: bool = Query(False),
    ):
        """
        Get artifact content by path components.

        Security: Path traversal prevention enforced.
        """
        artifact_repo: ArtifactRepository = app.state.artifact_repo

        # Resolve path safely
        path = artifact_repo.resolve_artifact_path(project_id, phase, run_id, artifact_path)
        if not path or not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found or access denied")

        # Determine content type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        if download:
            media_type = "application/octet-stream"
            disposition = f'attachment; filename="{path.name}"'
        else:
            media_type = mime_type
            disposition = f'inline; filename="{path.name}"'

        return FileResponse(
            path=path,
            media_type=media_type,
            headers={"Content-Disposition": disposition},
        )

    # -------------------------------------------------------------------------
    # Plot data endpoints
    # -------------------------------------------------------------------------

    @app.get(
        "/api/runs/{run_key}/plots/{plot_key}",
        tags=["Plots"],
    )
    async def get_plot_data(
        run_key: str,
        plot_key: str,
    ):
        """
        Get plot JSON data for interactive chart rendering.

        Parameters
        ----------
        run_key : str
            Run key (project:phase:run_id)
        plot_key : str
            Plot key (e.g., 'roc_averaged', 'pr_averaged', 'roc_inference')

        Returns
        -------
        JSONResponse
            Plot curve data for rendering in the UI
        """
        artifact_repo: ArtifactRepository = app.state.artifact_repo

        # Parse run key
        parts = run_key.split(":")
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail="Invalid run_key format")

        project_id, phase, run_id = parts

        # Map plot key to file path
        plot_files = {
            "roc_averaged": "plots/roc_averaged.json",
            "pr_averaged": "plots/pr_averaged.json",
            "roc_by_fold": "plots/roc_by_fold.json",
            "pr_by_fold": "plots/pr_by_fold.json",
            "roc_inference": "plots/roc_inference.json",
            "pr_inference": "plots/pr_inference.json",
        }

        if plot_key not in plot_files:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plot_key. Must be one of: {', '.join(plot_files.keys())}"
            )

        # Resolve path
        path = artifact_repo.resolve_artifact_path(project_id, phase, run_id, plot_files[plot_key])
        if not path or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Plot data not available for this run (key: {plot_key})"
            )

        return FileResponse(
            path=path,
            media_type="application/json",
        )

    @app.get(
        "/api/projects/{project_id}/runs/{phase}/{run_id}/plots/{plot_key}",
        tags=["Plots"],
    )
    async def get_plot_data_by_path(
        project_id: str,
        phase: str,
        run_id: str,
        plot_key: str,
    ):
        """
        Get plot JSON data by path components.

        This is an alternative endpoint with more discoverable URLs.
        """
        run_key = f"{project_id}:{phase}:{run_id}"
        return await get_plot_data(run_key, plot_key)

    # -------------------------------------------------------------------------
    # Comment endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/comments", response_model=PaginatedResponse, tags=["Comments"])
    async def list_comments(
        scope_type: str = Query(..., description="Scope type: project, run, or artifact"),
        scope_id: str = Query(..., description="Scope identifier"),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
    ):
        """List comments for a scope."""
        comment_repo: CommentRepository = app.state.comment_repo
        comments, total = comment_repo.list_comments(
            scope_type=scope_type,
            scope_id=scope_id,
            page=page,
            page_size=page_size,
        )

        return PaginatedResponse(
            items=[c.model_dump() for c in comments],
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total,
            has_prev=page > 1,
        )

    @app.post("/api/comments", response_model=Comment, tags=["Comments"])
    async def create_comment(data: CommentCreate):
        """Create a new comment."""
        comment_repo: CommentRepository = app.state.comment_repo

        # Validate scope_type
        if data.scope_type not in ["project", "run", "artifact"]:
            raise HTTPException(
                status_code=400,
                detail="scope_type must be one of: project, run, artifact",
            )

        return comment_repo.create_comment(data)

    @app.delete("/api/comments/{comment_id}", tags=["Comments"])
    async def delete_comment(comment_id: int):
        """Delete a comment."""
        comment_repo: CommentRepository = app.state.comment_repo
        if not comment_repo.delete_comment(comment_id):
            raise HTTPException(status_code=404, detail="Comment not found")
        return {"status": "deleted"}

    # -------------------------------------------------------------------------
    # Review endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/reviews", response_model=PaginatedResponse, tags=["Reviews"])
    async def list_reviews(
        scope_type: str = Query(..., description="Scope type: project or run"),
        scope_id: str = Query(..., description="Scope identifier"),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
    ):
        """List reviews for a scope."""
        review_repo: ReviewRepository = app.state.review_repo
        reviews, total = review_repo.list_reviews(
            scope_type=scope_type,
            scope_id=scope_id,
            page=page,
            page_size=page_size,
        )

        return PaginatedResponse(
            items=[r.model_dump() for r in reviews],
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total,
            has_prev=page > 1,
        )

    @app.post("/api/reviews", response_model=Review, tags=["Reviews"])
    async def create_review(data: ReviewCreate):
        """Create a new review."""
        review_repo: ReviewRepository = app.state.review_repo

        # Validate scope_type
        if data.scope_type not in ["project", "run"]:
            raise HTTPException(
                status_code=400,
                detail="scope_type must be one of: project, run",
            )

        return review_repo.create_review(data)

    @app.patch("/api/reviews/{review_id}", response_model=Review, tags=["Reviews"])
    async def update_review(
        review_id: int,
        status: ReviewStatus,
        notes: Optional[str] = None,
    ):
        """Update review status."""
        review_repo: ReviewRepository = app.state.review_repo
        review = review_repo.update_review_status(review_id, status, notes)
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")
        return review

    # -------------------------------------------------------------------------
    # Reindex endpoint
    # -------------------------------------------------------------------------

    @app.post("/api/reindex", tags=["System"])
    async def reindex():
        """Trigger a reindex of projects."""
        config: UIConfig = app.state.config
        if config.storage_mode == StorageMode.LOCAL:
            local_repo = app.state.project_repo
            if hasattr(local_repo, "refresh"):
                local_repo.refresh()
        return {"status": "reindexed"}


def mount_static(app: FastAPI, static_dir: Path):
    """
    Mount static files for serving the frontend.

    Parameters
    ----------
    app : FastAPI
        Application to mount on
    static_dir : Path
        Directory containing built frontend
    """
    if static_dir.is_dir():
        # Mount at root, but after API routes
        app.mount("/", SPAStaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Mounted static files from {static_dir}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
