"""Statistics subsystem for publication-ready statistical analysis and reporting.

This module provides a comprehensive statistical analysis framework for multiclass
and pairwise comparisons, with support for:

- Normality testing (Shapiro-Wilk)
- Parametric tests (Welch t-test, ANOVA, Tukey HSD)
- Nonparametric tests (Kruskal-Wallis, Dunn post-hoc)
- Effect size calculations (Cohen's d, Cliff's delta, log2FC)
- Publication-ready Excel workbooks with multiple sheets
- Statistical visualizations (boxplots, volcano plots, heatmaps)

Public API:
-----------
from classiflow.stats import run_stats, run_visualizations, StatsConfig

# Run complete statistical analysis
results = run_stats(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="derived/stats_results",
    alpha=0.05
)

# Generate visualizations
run_visualizations(
    data_csv="data.csv",
    label_col="diagnosis",
    stats_dir="derived/stats_results",
    outdir="derived/viz"
)
"""

from classiflow.stats.api import run_stats, run_visualizations
from classiflow.stats.config import StatsConfig, VizConfig

__all__ = ["run_stats", "run_visualizations", "StatsConfig", "VizConfig"]
