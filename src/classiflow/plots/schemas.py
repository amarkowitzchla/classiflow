"""Plot data JSON schemas for curve serialization.

This module defines Pydantic models for serializing ROC and PR curve data
to JSON format, enabling interactive visualization in the UI without
relying on static PNG images.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PlotType(str, Enum):
    """Type of plot curve."""

    ROC = "roc"
    PR = "pr"


class PlotScope(str, Enum):
    """Scope of the plot data."""

    AVERAGED = "averaged"
    FOLD = "fold"
    INFERENCE = "inference"


class TaskType(str, Enum):
    """Classification task type."""

    BINARY = "binary"
    MULTICLASS = "multiclass"


class CurveData(BaseModel):
    """Individual curve data within a plot."""

    label: str = Field(
        ...,
        description="Curve label: 'macro', 'micro', 'weighted', class name, or 'overall'"
    )
    x: list[float] = Field(
        ...,
        description="X-axis values (FPR for ROC, Recall for PR)"
    )
    y: list[float] = Field(
        ...,
        description="Y-axis values (TPR for ROC, Precision for PR)"
    )
    thresholds: Optional[list[float]] = Field(
        default=None,
        description="Decision thresholds corresponding to each point (optional)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "label": "micro",
                "x": [0.0, 0.1, 0.2, 0.5, 1.0],
                "y": [0.0, 0.4, 0.7, 0.9, 1.0],
                "thresholds": [1.0, 0.8, 0.6, 0.4, 0.0],
            }
        }


class PlotSummary(BaseModel):
    """Summary statistics for the plot."""

    auc: Optional[dict[str, float]] = Field(
        default=None,
        description="AUC values keyed by label (e.g., {'macro': 0.93, 'micro': 0.95})"
    )
    ap: Optional[dict[str, float]] = Field(
        default=None,
        description="Average Precision values for PR curves"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "auc": {"macro": 0.93, "micro": 0.95, "ClassA": 0.92},
            }
        }


class PlotMetadata(BaseModel):
    """Metadata about how the plot was generated."""

    generated_at: datetime = Field(
        ...,
        description="ISO8601 timestamp when the data was generated"
    )
    source: str = Field(
        ...,
        description="Source of the data: 'metrics.json', 'predictions.csv', 'internal'"
    )
    classiflow_version: str = Field(
        ...,
        description="Version of Classiflow that generated the data"
    )
    run_id: str = Field(
        ...,
        description="Run ID this plot belongs to"
    )
    fold: Optional[int] = Field(
        default=None,
        description="Fold number (1-indexed) if scope is 'fold'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "generated_at": "2024-01-15T10:30:00",
                "source": "internal",
                "classiflow_version": "0.1.0",
                "run_id": "abc123",
                "fold": 1,
            }
        }


class PlotCurve(BaseModel):
    """Complete plot curve data for serialization to JSON.

    This is the main data structure for ROC and PR curve JSON files.
    """

    plot_type: PlotType = Field(
        ...,
        description="Type of plot: 'roc' or 'pr'"
    )
    scope: PlotScope = Field(
        ...,
        description="Scope of the data: 'averaged', 'fold', or 'inference'"
    )
    task: TaskType = Field(
        ...,
        description="Classification task type: 'binary' or 'multiclass'"
    )
    labels: list[str] = Field(
        ...,
        description="Class labels in order"
    )
    curves: list[CurveData] = Field(
        ...,
        description="Individual curves to plot"
    )
    summary: PlotSummary = Field(
        default_factory=PlotSummary,
        description="Summary metrics (AUC, AP)"
    )
    metadata: PlotMetadata = Field(
        ...,
        description="Generation metadata"
    )

    # For averaged plots with standard deviation bands
    std_band: Optional[dict[str, Any]] = Field(
        default=None,
        description="Standard deviation band for averaged plots: {x: [...], y_upper: [...], y_lower: [...]}"
    )

    # For fold-level plots, include individual fold data
    fold_curves: Optional[list[CurveData]] = Field(
        default=None,
        description="Individual fold curves for averaged plots"
    )
    fold_metrics: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="Per-fold metric values (e.g., AUC per fold)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "plot_type": "roc",
                "scope": "averaged",
                "task": "multiclass",
                "labels": ["ClassA", "ClassB", "ClassC"],
                "curves": [
                    {"label": "mean", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.8, 1.0]},
                    {"label": "micro", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.85, 1.0]},
                ],
                "summary": {"auc": {"mean": 0.93, "micro": 0.95}},
                "metadata": {
                    "generated_at": "2024-01-15T10:30:00",
                    "source": "internal",
                    "classiflow_version": "0.1.0",
                    "run_id": "abc123",
                },
            }
        }


class PlotManifest(BaseModel):
    """Manifest listing available plot data files for a run.

    This file helps the UI discover which JSON plot files are available
    and provides fallback PNG paths for backwards compatibility.
    """

    available: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of plot key to relative JSON file path"
    )
    fallback_pngs: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of plot key to fallback PNG file path"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the manifest was generated"
    )
    classiflow_version: str = Field(
        default="",
        description="Classiflow version that generated the manifest"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "available": {
                    "roc_averaged": "plots/roc_averaged.json",
                    "pr_averaged": "plots/pr_averaged.json",
                    "roc_by_fold": "plots/roc_by_fold.json",
                    "pr_by_fold": "plots/pr_by_fold.json",
                },
                "fallback_pngs": {
                    "roc_averaged": "averaged_roc.png",
                    "pr_averaged": "averaged_pr.png",
                },
                "generated_at": "2024-01-15T10:30:00",
                "classiflow_version": "0.1.0",
            }
        }


# Plot key constants for standardized naming
class PlotKey:
    """Standard keys for plot types in the manifest."""

    ROC_AVERAGED = "roc_averaged"
    PR_AVERAGED = "pr_averaged"
    ROC_BY_FOLD = "roc_by_fold"
    PR_BY_FOLD = "pr_by_fold"
    ROC_INFERENCE = "roc_inference"
    PR_INFERENCE = "pr_inference"
