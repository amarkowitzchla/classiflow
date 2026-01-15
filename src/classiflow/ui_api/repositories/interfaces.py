"""Abstract repository interfaces for storage abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from classiflow.ui_api.models import (
    ProjectCard,
    ProjectDashboard,
    RunBrief,
    RunDetail,
    Artifact,
    Comment,
    CommentCreate,
    Review,
    ReviewCreate,
    ReviewStatus,
)


class ProjectRepository(ABC):
    """Interface for project data access."""

    @abstractmethod
    def list_projects(
        self,
        query: Optional[str] = None,
        mode: Optional[str] = None,
        owner: Optional[str] = None,
        updated_after: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ProjectCard], int]:
        """
        List projects with optional filtering.

        Returns tuple of (projects, total_count).
        """
        pass

    @abstractmethod
    def get_project(self, project_id: str) -> Optional[ProjectDashboard]:
        """Get full project dashboard by ID."""
        pass

    @abstractmethod
    def get_project_runs(self, project_id: str) -> dict[str, list[RunBrief]]:
        """Get runs grouped by phase for a project."""
        pass


class RunRepository(ABC):
    """Interface for run data access."""

    @abstractmethod
    def get_run(self, run_key: str) -> Optional[RunDetail]:
        """Get run detail by composite key (project:phase:run_id)."""
        pass

    @abstractmethod
    def list_runs(
        self,
        project_id: Optional[str] = None,
        phase: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[RunBrief], int]:
        """List runs with optional filtering."""
        pass


class ArtifactRepository(ABC):
    """Interface for artifact data access."""

    @abstractmethod
    def list_artifacts(
        self,
        run_key: str,
        kind: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Artifact], int]:
        """List artifacts for a run with optional kind filter."""
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact metadata by ID."""
        pass

    @abstractmethod
    def get_artifact_path(self, artifact_id: str) -> Optional[Path]:
        """Get the filesystem path for an artifact (for serving)."""
        pass

    @abstractmethod
    def resolve_artifact_path(
        self,
        project_id: str,
        phase: str,
        run_id: str,
        relative_path: str,
    ) -> Optional[Path]:
        """
        Safely resolve an artifact path from components.

        Returns None if path is invalid or escapes boundaries.
        """
        pass


class CommentRepository(ABC):
    """Interface for comment persistence."""

    @abstractmethod
    def list_comments(
        self,
        scope_type: str,
        scope_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Comment], int]:
        """List comments for a scope."""
        pass

    @abstractmethod
    def create_comment(self, data: CommentCreate) -> Comment:
        """Create a new comment."""
        pass

    @abstractmethod
    def get_comment(self, comment_id: int) -> Optional[Comment]:
        """Get comment by ID."""
        pass

    @abstractmethod
    def delete_comment(self, comment_id: int) -> bool:
        """Delete a comment. Returns True if deleted."""
        pass


class ReviewRepository(ABC):
    """Interface for review persistence."""

    @abstractmethod
    def list_reviews(
        self,
        scope_type: str,
        scope_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Review], int]:
        """List reviews for a scope."""
        pass

    @abstractmethod
    def create_review(self, data: ReviewCreate) -> Review:
        """Create a new review."""
        pass

    @abstractmethod
    def get_review(self, review_id: int) -> Optional[Review]:
        """Get review by ID."""
        pass

    @abstractmethod
    def update_review_status(
        self,
        review_id: int,
        status: ReviewStatus,
        notes: Optional[str] = None,
    ) -> Optional[Review]:
        """Update review status."""
        pass

    @abstractmethod
    def get_latest_review(
        self,
        scope_type: str,
        scope_id: str,
    ) -> Optional[Review]:
        """Get the most recent review for a scope."""
        pass
