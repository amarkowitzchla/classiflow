"""Plotting utilities for ROC curves, confusion matrices, etc."""

from classiflow.plots.hierarchical import (
    plot_roc_curve,
    plot_pr_curve,
    plot_averaged_roc_curves,
    plot_averaged_pr_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    extract_feature_importance_mlp,
)
from classiflow.plots.data_export import (
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_averaged_roc_data,
    compute_averaged_pr_data,
    save_plot_data,
    create_plot_manifest,
    generate_technical_validation_plots,
    generate_inference_plots,
)
from classiflow.plots.schemas import (
    PlotCurve,
    PlotManifest,
    PlotKey,
    PlotType,
    PlotScope,
    TaskType,
    CurveData,
    PlotSummary,
    PlotMetadata,
)

__all__ = [
    # PNG plotting functions
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_averaged_roc_curves",
    "plot_averaged_pr_curves",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "extract_feature_importance_mlp",
    # JSON data export functions
    "compute_roc_curve_data",
    "compute_pr_curve_data",
    "compute_averaged_roc_data",
    "compute_averaged_pr_data",
    "save_plot_data",
    "create_plot_manifest",
    "generate_technical_validation_plots",
    "generate_inference_plots",
    # Schema classes
    "PlotCurve",
    "PlotManifest",
    "PlotKey",
    "PlotType",
    "PlotScope",
    "TaskType",
    "CurveData",
    "PlotSummary",
    "PlotMetadata",
]
