"""Plotting utilities for ROC curves, confusion matrices, etc."""

from classiflow.plots.data_export import (
    compute_averaged_pr_data,
    compute_averaged_roc_data,
    compute_pr_curve_data,
    compute_roc_curve_data,
    create_plot_manifest,
    generate_inference_plots,
    generate_technical_validation_plots,
    save_plot_data,
)
from classiflow.plots.hierarchical import (
    extract_feature_importance_mlp,
    plot_averaged_pr_curves,
    plot_averaged_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from classiflow.plots.schemas import (
    CurveData,
    PlotCurve,
    PlotKey,
    PlotManifest,
    PlotMetadata,
    PlotScope,
    PlotSummary,
    PlotType,
    TaskType,
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
