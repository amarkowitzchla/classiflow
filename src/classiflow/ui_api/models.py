"""Pydantic models for UI API responses (Gold layer)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    """Standard project phases."""

    FEASIBILITY = "feasibility"
    TECHNICAL_VALIDATION = "technical_validation"
    INDEPENDENT_TEST = "independent_test"
    FINAL_MODEL = "final_model"


class DecisionBadge(str, Enum):
    """Promotion gate decision status."""

    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    OVERRIDE = "OVERRIDE"


class ArtifactKind(str, Enum):
    """Categories of artifacts."""

    IMAGE = "image"
    REPORT = "report"
    METRICS = "metrics"
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    OTHER = "other"


class ReviewStatus(str, Enum):
    """Review status options."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    needs_changes = "needs_changes"


class GateStatus(str, Enum):
    """Status of a single promotion gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"  # Not yet run or no data


# ---------------------------------------------------------------------------
# Run-level models
# ---------------------------------------------------------------------------


class MetricsSummary(BaseModel):
    """Summary metrics for a run."""

    primary: dict[str, float] = Field(default_factory=dict, description="Primary metrics")
    per_fold: dict[str, list[float]] = Field(
        default_factory=dict, description="Metrics per fold (for validation phases)"
    )
    per_class: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-class metrics"
    )
    confusion_matrix: Optional[dict[str, Any]] = Field(
        default=None, description="Confusion matrix data"
    )
    roc_auc: Optional[dict[str, Any]] = Field(default=None, description="ROC AUC data")


class RunBrief(BaseModel):
    """Brief run info for listings."""

    run_key: str = Field(..., description="Globally unique run key: project:phase:run_id")
    run_id: str = Field(..., description="Short run identifier (directory name)")
    phase: str = Field(..., description="Phase name")
    created_at: Optional[datetime] = Field(default=None, description="Run timestamp")
    task_type: Optional[str] = Field(default=None, description="Task type: binary/meta/hierarchical")
    headline_metrics: dict[str, float] = Field(
        default_factory=dict, description="Key metrics for display"
    )


class RunDetail(BaseModel):
    """Full run details."""

    run_key: str = Field(..., description="Globally unique run key")
    run_id: str = Field(..., description="Short run identifier")
    project_id: str = Field(..., description="Parent project ID")
    phase: str = Field(..., description="Phase name")
    created_at: Optional[datetime] = Field(default=None, description="Run timestamp")
    task_type: Optional[str] = Field(default=None, description="Task type")
    config: dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    metrics: MetricsSummary = Field(default_factory=MetricsSummary, description="Metrics summary")
    feature_count: int = Field(default=0, description="Number of features")
    feature_list: list[str] = Field(default_factory=list, description="Feature names")
    lineage: Optional[dict[str, Any]] = Field(default=None, description="Lineage metadata")
    artifact_count: int = Field(default=0, description="Number of artifacts")
    artifacts: list[Artifact] = Field(default_factory=list, description="Run artifacts")


# ---------------------------------------------------------------------------
# Artifact models
# ---------------------------------------------------------------------------


class Artifact(BaseModel):
    """Artifact metadata and URLs."""

    artifact_id: str = Field(..., description="Stable artifact identifier")
    title: str = Field(..., description="Display title")
    relative_path: str = Field(..., description="Path relative to run directory")
    kind: ArtifactKind = Field(default=ArtifactKind.OTHER, description="Artifact category")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    size_bytes: Optional[int] = Field(default=None, description="File size")
    created_at: Optional[datetime] = Field(default=None, description="File modification time")
    run_key: str = Field(..., description="Parent run key")
    phase: str = Field(..., description="Phase name")
    is_viewable: bool = Field(default=False, description="Can be viewed inline")
    view_url: Optional[str] = Field(default=None, description="URL for viewing")
    download_url: Optional[str] = Field(default=None, description="URL for downloading")


# ---------------------------------------------------------------------------
# Project-level models
# ---------------------------------------------------------------------------


class RegistrySummary(BaseModel):
    """Summary of registered datasets."""

    train: Optional[dict[str, Any]] = Field(default=None, description="Train dataset info")
    test: Optional[dict[str, Any]] = Field(default=None, description="Test dataset info")
    thresholds: Optional[dict[str, Any]] = Field(default=None, description="Promotion thresholds")


class GateCheck(BaseModel):
    """Result of a single threshold check."""

    metric: str = Field(..., description="Metric name as specified in thresholds")
    threshold: float = Field(..., description="Required threshold")
    actual: Optional[float] = Field(default=None, description="Actual value")
    passed: bool = Field(default=False, description="Check passed")
    check_type: str = Field(default="required", description="Check type: required, stability_std, stability_pass_rate, safety")


class GateResult(BaseModel):
    """Result of evaluating a promotion gate for one phase."""

    phase: str = Field(..., description="Phase name")
    phase_label: str = Field(default="", description="Human-readable phase name")
    passed: bool = Field(default=False, description="Gate passed")
    run_id: Optional[str] = Field(default=None, description="Run ID evaluated")
    run_key: Optional[str] = Field(default=None, description="Full run key")
    checks: list[GateCheck] = Field(default_factory=list, description="Individual threshold checks")
    metrics_available: dict[str, float] = Field(
        default_factory=dict, description="All metrics available from this run"
    )


class PromotionSummary(BaseModel):
    """Promotion gate summary."""

    decision: DecisionBadge = Field(default=DecisionBadge.PENDING, description="Decision status")
    timestamp: Optional[datetime] = Field(default=None, description="Decision timestamp")
    technical_run: Optional[str] = Field(default=None, description="Technical validation run ID")
    test_run: Optional[str] = Field(default=None, description="Independent test run ID")
    reasons: list[str] = Field(default_factory=list, description="Decision reasons")
    override_enabled: bool = Field(default=False, description="Override enabled")
    override_comment: Optional[str] = Field(default=None, description="Override comment")
    override_approver: Optional[str] = Field(default=None, description="Override approver")
    # Detailed gate results per phase
    gates: dict[str, GateResult] = Field(
        default_factory=dict, description="Gate results by phase"
    )


class ProjectCard(BaseModel):
    """Project summary for listings."""

    id: str = Field(..., description="Project directory name")
    name: str = Field(..., description="Human-readable project name")
    description: Optional[str] = Field(default=None, description="Project description")
    owner: Optional[str] = Field(default=None, description="Project owner")
    task_mode: Optional[str] = Field(default=None, description="Task mode: binary/meta/hierarchical")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    phases_present: list[str] = Field(default_factory=list, description="Available phases")
    decision_badge: DecisionBadge = Field(
        default=DecisionBadge.PENDING, description="Promotion decision"
    )
    gate_status: dict[str, GateStatus] = Field(
        default_factory=dict,
        description="Per-gate status: technical_validation, independent_test"
    )
    latest_runs_by_phase: dict[str, RunBrief] = Field(
        default_factory=dict, description="Latest run per phase"
    )
    headline_metrics: dict[str, float] = Field(
        default_factory=dict, description="Key metrics for display"
    )
    run_count: int = Field(default=0, description="Total run count")


class ProjectDashboard(BaseModel):
    """Full project dashboard data."""

    id: str = Field(..., description="Project directory name")
    name: str = Field(..., description="Human-readable project name")
    description: Optional[str] = Field(default=None)
    owner: Optional[str] = Field(default=None)
    task_mode: Optional[str] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    registry: RegistrySummary = Field(
        default_factory=RegistrySummary, description="Registry summary"
    )
    promotion: PromotionSummary = Field(
        default_factory=PromotionSummary, description="Promotion summary"
    )
    phases: dict[str, list[RunBrief]] = Field(
        default_factory=dict, description="Runs grouped by phase"
    )
    artifact_highlights: list[Artifact] = Field(
        default_factory=list, description="Featured artifacts"
    )


# ---------------------------------------------------------------------------
# Collaboration models
# ---------------------------------------------------------------------------


class CommentCreate(BaseModel):
    """Request body for creating a comment."""

    scope_type: str = Field(..., description="Scope type: project, run, or artifact")
    scope_id: str = Field(..., description="Scope identifier")
    author: str = Field(..., description="Comment author")
    content: str = Field(..., description="Comment content (markdown)")


class Comment(BaseModel):
    """Comment on a project, run, or artifact."""

    id: int = Field(..., description="Comment ID")
    scope_type: str = Field(..., description="Scope type: project, run, or artifact")
    scope_id: str = Field(..., description="Scope identifier")
    author: str = Field(..., description="Comment author")
    content: str = Field(..., description="Comment content (markdown)")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")


class ReviewCreate(BaseModel):
    """Request body for creating a review."""

    scope_type: str = Field(..., description="Scope type: project or run")
    scope_id: str = Field(..., description="Scope identifier")
    reviewer: str = Field(..., description="Reviewer name")
    status: ReviewStatus = Field(..., description="Review status")
    notes: Optional[str] = Field(default=None, description="Review notes")


class Review(BaseModel):
    """Review of a project or run."""

    id: int = Field(..., description="Review ID")
    scope_type: str = Field(..., description="Scope type: project or run")
    scope_id: str = Field(..., description="Scope identifier")
    reviewer: str = Field(..., description="Reviewer name")
    status: ReviewStatus = Field(..., description="Review status")
    notes: Optional[str] = Field(default=None, description="Review notes")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")


# ---------------------------------------------------------------------------
# API response wrappers
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service status")
    storage_mode: str = Field(..., description="Storage mode: local, postgres, databricks")
    projects_root: str = Field(..., description="Projects root directory")
    db_path: Optional[str] = Field(default=None, description="SQLite database path")
    project_count: int = Field(default=0, description="Number of discovered projects")
    index_status: str = Field(default="ready", description="Index status")
    version: str = Field(..., description="Classiflow version")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""

    items: list[Any] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)
    has_next: bool = Field(default=False)
    has_prev: bool = Field(default=False)
