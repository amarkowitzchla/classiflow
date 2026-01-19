"""
Publication-Ready Plotting Functions for SMOTE Comparison

This module provides high-quality visualization functions for comparing SMOTE
and no-SMOTE model performance.

Plot Types:
- Delta bar charts (SMOTE - no-SMOTE performance differences)
- Identity scatter plots (parity plots)
- Distribution comparisons (violin/box plots)
- Per-fold trajectories
- Statistical significance annotations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_delta_bars(
    deltas: Dict[str, float],
    pvalues: Optional[Dict[str, float]] = None,
    title: str = "Performance Difference (SMOTE - No-SMOTE)",
    xlabel: str = "Δ Metric",
    outfile: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    color_positive: str = "#2ecc71",
    color_negative: str = "#e74c3c",
    significance_stars: bool = True,
) -> plt.Figure:
    """
    Create horizontal bar chart of performance deltas.

    Args:
        deltas: Dictionary {metric_name: delta_value}
        pvalues: Optional p-values for significance annotation
        title: Plot title
        xlabel: X-axis label
        outfile: Path to save figure (optional)
        figsize: Figure size (width, height)
        color_positive: Color for positive deltas
        color_negative: Color for negative deltas
        significance_stars: Whether to add *** stars for significance

    Returns:
        Matplotlib figure
    """
    # Sort by delta magnitude
    sorted_items = sorted(deltas.items(), key=lambda x: x[1])
    metrics = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Determine colors
    colors = [color_positive if v > 0 else color_negative for v in values]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add zero line
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add significance stars
    if significance_stars and pvalues:
        for i, (metric, val) in enumerate(zip(metrics, values)):
            if metric in pvalues:
                p = pvalues[metric]
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"

                if stars:
                    # Position text at end of bar
                    x_pos = val + (0.005 if val > 0 else -0.005)
                    ha = "left" if val > 0 else "right"
                    ax.text(x_pos, i, stars, va='center', ha=ha, fontsize=12, fontweight='bold')

    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_positive, edgecolor='black', label='SMOTE Better'),
        Patch(facecolor=color_negative, edgecolor='black', label='No-SMOTE Better')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def plot_identity_scatter(
    no_smote_values: Union[pd.Series, np.ndarray, List],
    smote_values: Union[pd.Series, np.ndarray, List],
    metric_name: str = "Metric",
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 8),
    add_stats: bool = True,
) -> plt.Figure:
    """
    Create identity/parity scatter plot comparing SMOTE vs no-SMOTE.

    Args:
        no_smote_values: No-SMOTE metric values
        smote_values: SMOTE metric values
        metric_name: Name of metric being compared
        labels: Optional labels for each point
        title: Plot title (auto-generated if None)
        outfile: Path to save figure
        figsize: Figure size
        add_stats: Whether to add correlation and mean difference stats

    Returns:
        Matplotlib figure
    """
    no_smote = np.array(no_smote_values)
    smote = np.array(smote_values)

    # Remove NaN pairs
    valid = ~(np.isnan(no_smote) | np.isnan(smote))
    no_smote = no_smote[valid]
    smote = smote[valid]

    if len(no_smote) == 0:
        raise ValueError("No valid data points after removing NaNs")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(no_smote, smote, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2, label='Identity (y=x)')

    # Add labels if provided
    if labels:
        labels_valid = [labels[i] for i in range(len(labels)) if valid[i]]
        for i, label in enumerate(labels_valid):
            ax.annotate(
                label,
                (no_smote[i], smote[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=8,
                alpha=0.7
            )

    # Statistics
    if add_stats:
        correlation = np.corrcoef(no_smote, smote)[0, 1]
        mean_diff = np.mean(smote - no_smote)

        stats_text = f"r = {correlation:.3f}\nΔ = {mean_diff:+.4f}"
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11,
            fontweight='bold'
        )

    # Labels and formatting
    ax.set_xlabel(f'{metric_name} (No-SMOTE)', fontweight='bold')
    ax.set_ylabel(f'{metric_name} (SMOTE)', fontweight='bold')

    if title is None:
        title = f'{metric_name}: SMOTE vs No-SMOTE Comparison'
    ax.set_title(title, fontweight='bold', pad=15)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def plot_distribution_comparison(
    smote_data: pd.DataFrame,
    no_smote_data: pd.DataFrame,
    metric: str,
    plot_type: str = "violin",
    title: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Compare distributions of SMOTE vs no-SMOTE using violin or box plots.

    Args:
        smote_data: DataFrame with SMOTE results (must have 'fold' column)
        no_smote_data: DataFrame with no-SMOTE results
        metric: Metric column to compare
        plot_type: 'violin' or 'box'
        title: Plot title
        outfile: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Prepare data
    smote_vals = smote_data[metric].dropna()
    no_smote_vals = no_smote_data[metric].dropna()

    if smote_vals.empty and no_smote_vals.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available for this metric", va="center", ha="center", fontsize=12, fontweight="bold")
        ax.set_axis_off()
        if outfile:
            fig.savefig(outfile, dpi=300, bbox_inches='tight')
        return fig

    # Combine into long format
    combined = pd.DataFrame({
        'Value': pd.concat([no_smote_vals, smote_vals], ignore_index=True),
        'Method': ['No-SMOTE'] * len(no_smote_vals) + ['SMOTE'] * len(smote_vals)
    })

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "violin":
        sns.violinplot(
            data=combined,
            x='Method',
            y='Value',
            palette={"No-SMOTE": "#3498db", "SMOTE": "#e67e22"},
            ax=ax,
            inner="box",
            linewidth=2
        )
    else:  # box plot
        sns.boxplot(
            data=combined,
            x='Method',
            y='Value',
            palette={"No-SMOTE": "#3498db", "SMOTE": "#e67e22"},
            ax=ax,
            linewidth=2
        )

        # Add individual points
        sns.stripplot(
            data=combined,
            x='Method',
            y='Value',
            color='black',
            alpha=0.5,
            size=5,
            ax=ax
        )

    # Add mean markers
    means = combined.groupby('Method')['Value'].mean()
    for i, method in enumerate(['No-SMOTE', 'SMOTE']):
        ax.plot(i, means[method], marker='D', color='red', markersize=10, zorder=10,
                markeredgecolor='black', markeredgewidth=1.5, label='Mean' if i == 0 else '')

    # Labels
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_xlabel('')

    if title is None:
        title = f'{metric} Distribution: SMOTE vs No-SMOTE'
    ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def plot_fold_trajectories(
    smote_data: pd.DataFrame,
    no_smote_data: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot per-fold performance trajectories for SMOTE vs no-SMOTE.

    Args:
        smote_data: DataFrame with SMOTE results (must have 'fold' column)
        no_smote_data: DataFrame with no-SMOTE results
        metric: Metric to plot
        title: Plot title
        outfile: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Aggregate by fold
    smote_fold = smote_data.groupby('fold')[metric].mean()
    no_smote_fold = no_smote_data.groupby('fold')[metric].mean()

    # Ensure aligned folds
    folds = sorted(set(smote_fold.index) & set(no_smote_fold.index))
    smote_vals = [smote_fold[f] for f in folds]
    no_smote_vals = [no_smote_fold[f] for f in folds]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines
    ax.plot(folds, no_smote_vals, marker='o', markersize=10, linewidth=2.5,
            label='No-SMOTE', color='#3498db', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(folds, smote_vals, marker='s', markersize=10, linewidth=2.5,
            label='SMOTE', color='#e67e22', markeredgecolor='black', markeredgewidth=1.5)

    # Connect corresponding points with dashed lines
    for i, fold in enumerate(folds):
        ax.plot([fold, fold], [no_smote_vals[i], smote_vals[i]],
                linestyle='--', color='gray', alpha=0.5, linewidth=1)

    # Add mean lines
    smote_mean = np.mean(smote_vals)
    no_smote_mean = np.mean(no_smote_vals)

    ax.axhline(smote_mean, color='#e67e22', linestyle=':', linewidth=2, alpha=0.7,
               label=f'SMOTE Mean ({smote_mean:.3f})')
    ax.axhline(no_smote_mean, color='#3498db', linestyle=':', linewidth=2, alpha=0.7,
               label=f'No-SMOTE Mean ({no_smote_mean:.3f})')

    # Labels
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_xticks(folds)

    if title is None:
        title = f'{metric} Per-Fold Performance'
    ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def plot_metric_grid(
    smote_data: pd.DataFrame,
    no_smote_data: pd.DataFrame,
    metrics: List[str],
    outfile: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create grid of identity scatter plots for multiple metrics.

    Args:
        smote_data: DataFrame with SMOTE results
        no_smote_data: DataFrame with no-SMOTE results
        metrics: List of metrics to plot
        outfile: Path to save figure
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (6 * n_cols, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_metrics == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Get per-fold averages
        smote_fold = smote_data.groupby('fold')[metric].mean().values
        no_smote_fold = no_smote_data.groupby('fold')[metric].mean().values

        # Remove NaN pairs
        valid = ~(np.isnan(smote_fold) | np.isnan(no_smote_fold))
        smote_fold = smote_fold[valid]
        no_smote_fold = no_smote_fold[valid]

        if len(smote_fold) == 0:
            ax.text(0.5, 0.5, f'No valid data for {metric}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Scatter
        ax.scatter(no_smote_fold, smote_fold, s=100, alpha=0.7,
                   edgecolors='black', linewidth=1.5)

        # Identity line
        lims = [
            np.min([no_smote_fold.min(), smote_fold.min()]) * 0.95,
            np.max([no_smote_fold.max(), smote_fold.max()]) * 1.05,
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2)

        # Stats
        correlation = np.corrcoef(no_smote_fold, smote_fold)[0, 1]
        mean_diff = np.mean(smote_fold - no_smote_fold)

        stats_text = f"r={correlation:.2f}\nΔ={mean_diff:+.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)

        # Labels
        ax.set_xlabel(f'{metric} (No-SMOTE)', fontsize=10)
        ax.set_ylabel(f'{metric} (SMOTE)', fontsize=10)
        ax.set_title(metric, fontweight='bold', fontsize=11)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def plot_per_task_comparison(
    smote_data: pd.DataFrame,
    no_smote_data: pd.DataFrame,
    metric: str,
    task_col: str = "task",
    title: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create grouped bar chart comparing SMOTE vs no-SMOTE per task.

    Args:
        smote_data: DataFrame with SMOTE results
        no_smote_data: DataFrame with no-SMOTE results
        metric: Metric to compare
        task_col: Column name for tasks
        title: Plot title
        outfile: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Aggregate by task
    smote_task = smote_data.groupby(task_col)[metric].mean()
    no_smote_task = no_smote_data.groupby(task_col)[metric].mean()

    # Align tasks
    tasks = sorted(set(smote_task.index) & set(no_smote_task.index))
    smote_vals = [smote_task[t] for t in tasks]
    no_smote_vals = [no_smote_task[t] for t in tasks]

    # Create figure
    if figsize is None:
        figsize = (max(10, len(tasks) * 0.8), 6)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_smote_vals, width, label='No-SMOTE',
                   color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, smote_vals, width, label='SMOTE',
                   color='#e67e22', edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # Labels
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')

    if title is None:
        title = f'{metric} by Task: SMOTE vs No-SMOTE'
    ax.set_title(title, fontweight='bold', pad=15)

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    return fig


def create_all_plots(
    smote_data: pd.DataFrame,
    no_smote_data: pd.DataFrame,
    metrics: List[str],
    outdir: Union[str, Path],
    deltas: Optional[Dict[str, float]] = None,
    pvalues: Optional[Dict[str, float]] = None,
    prefix: str = "smote_comparison",
) -> Dict[str, Path]:
    """
    Generate all SMOTE comparison plots and save to directory.

    Args:
        smote_data: DataFrame with SMOTE results
        no_smote_data: DataFrame with no-SMOTE results
        metrics: List of metrics to plot
        outdir: Output directory
        deltas: Optional pre-computed deltas
        pvalues: Optional pre-computed p-values
        prefix: File prefix

    Returns:
        Dictionary mapping plot type to file path
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    created_files = {}

    # 1. Delta bar chart
    if deltas and pvalues:
        plot_delta_bars(
            deltas, pvalues,
            outfile=outdir / f"{prefix}_delta_bars.png"
        )
        created_files["delta_bars"] = outdir / f"{prefix}_delta_bars.png"

    # 2. Metric grid (identity scatter for all metrics)
    plot_metric_grid(
        smote_data, no_smote_data, metrics,
        outfile=outdir / f"{prefix}_identity_grid.png"
    )
    created_files["identity_grid"] = outdir / f"{prefix}_identity_grid.png"

    # 3. Distribution comparisons (violin plots)
    for metric in metrics:
        if metric in smote_data.columns and metric in no_smote_data.columns:
            plot_distribution_comparison(
                smote_data, no_smote_data, metric,
                plot_type="violin",
                outfile=outdir / f"{prefix}_dist_{metric}.png"
            )
            created_files[f"dist_{metric}"] = outdir / f"{prefix}_dist_{metric}.png"

    # 4. Fold trajectories
    for metric in metrics[:3]:  # Limit to top 3 metrics
        if metric in smote_data.columns and metric in no_smote_data.columns:
            plot_fold_trajectories(
                smote_data, no_smote_data, metric,
                outfile=outdir / f"{prefix}_trajectory_{metric}.png"
            )
            created_files[f"trajectory_{metric}"] = outdir / f"{prefix}_trajectory_{metric}.png"

    # 5. Per-task comparison (if task column exists)
    if "task" in smote_data.columns:
        for metric in metrics[:2]:  # Top 2 metrics
            if metric in smote_data.columns and metric in no_smote_data.columns:
                plot_per_task_comparison(
                    smote_data, no_smote_data, metric,
                    outfile=outdir / f"{prefix}_per_task_{metric}.png"
                )
                created_files[f"per_task_{metric}"] = outdir / f"{prefix}_per_task_{metric}.png"

    return created_files
