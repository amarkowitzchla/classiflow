"""Configuration dataclasses for stats subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class StatsConfig:
    """Configuration for statistical analysis.

    Attributes:
        data_csv: Path to input CSV with features and labels
        label_col: Name of the label/class column
        outdir: Output directory for results
        classes: Optional subset/order of classes to include (None = all)
        alpha: Significance threshold (default: 0.05)
        min_n: Minimum n per class to run Shapiro-Wilk (default: 3)
        dunn_adjust: P-value adjustment method for Dunn test
            Options: holm, bonferroni, fdr_bh, fdr_by, sidak
        feature_whitelist: Optional list of features to include (None = all numeric)
        feature_blacklist: Optional list of features to exclude
        top_n_features: Number of top features to include in summary (default: 30)
        write_legacy_csv: Whether to write legacy CSV outputs (default: True)
        write_legacy_xlsx: Whether to write legacy stats_results.xlsx (default: True)
    """

    data_csv: Path
    label_col: str
    outdir: Path
    classes: Optional[List[str]] = None
    alpha: float = 0.05
    min_n: int = 3
    dunn_adjust: str = "holm"
    feature_whitelist: Optional[List[str]] = None
    feature_blacklist: Optional[List[str]] = None
    top_n_features: int = 30
    write_legacy_csv: bool = True
    write_legacy_xlsx: bool = True

    def __post_init__(self):
        """Validate configuration."""
        self.data_csv = Path(self.data_csv)
        self.outdir = Path(self.outdir)

        if not self.data_csv.exists():
            raise FileNotFoundError(f"Data CSV not found: {self.data_csv}")

        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError(f"Alpha must be in (0, 1), got {self.alpha}")

        if self.min_n < 2:
            raise ValueError(f"min_n must be >= 2, got {self.min_n}")

        valid_adjustments = ["holm", "bonferroni", "fdr_bh", "fdr_by", "sidak"]
        if self.dunn_adjust not in valid_adjustments:
            raise ValueError(
                f"dunn_adjust must be one of {valid_adjustments}, got {self.dunn_adjust}"
            )


@dataclass
class VizConfig:
    """Configuration for statistical visualizations.

    Attributes:
        data_csv: Path to input CSV with features and labels
        label_col: Name of the label/class column
        outdir: Output directory for visualizations
        stats_dir: Optional directory with stats results for volcano plots
        classes: Optional subset/order of classes to include
        alpha: Significance threshold for volcano plots (default: 0.05)
        fc_thresh: |log2FC| threshold to flag in volcano plots (default: 1.0)
        fc_center: Center measure for fold-change (mean or median)
        fc_eps: Pseudocount for fold-change ratios (default: 1e-9)
        label_topk: Number of top features to annotate on volcano plots
        boxplot_ncols: Number of columns for boxplot grid
        heatmap_topn: Number of top features for heatmap (0 = skip)
        fig_dpi: Figure DPI (default: 160)
        point_size: Scatter point size (default: 48.0)
        alpha_points: Scatter point transparency (default: 0.9)
    """

    data_csv: Path
    label_col: str
    outdir: Path
    stats_dir: Optional[Path] = None
    classes: Optional[List[str]] = None
    alpha: float = 0.05
    fc_thresh: float = 1.0
    fc_center: str = "median"
    fc_eps: float = 1e-9
    label_topk: int = 12
    boxplot_ncols: int = 3
    heatmap_topn: int = 30
    fig_dpi: int = 160
    point_size: float = 48.0
    alpha_points: float = 0.9

    def __post_init__(self):
        """Validate configuration."""
        self.data_csv = Path(self.data_csv)
        self.outdir = Path(self.outdir)

        if self.stats_dir is not None:
            self.stats_dir = Path(self.stats_dir)

        if not self.data_csv.exists():
            raise FileNotFoundError(f"Data CSV not found: {self.data_csv}")

        if self.fc_center not in ["mean", "median"]:
            raise ValueError(f"fc_center must be 'mean' or 'median', got {self.fc_center}")
