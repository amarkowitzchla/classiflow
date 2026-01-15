"""Statistical visualizations (boxplots, volcano, fold-change, heatmaps)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from classiflow.stats.config import VizConfig
from classiflow.stats.effects import log2_fold_change

# Suppress seaborn/pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def choose_colors(classes: List[str]) -> Dict[str, tuple]:
    """Map classes to colors.

    Uses fixed order: blue, red, yellow, green for first 4 classes,
    then tab20 colormap for additional classes.

    Args:
        classes: List of class labels

    Returns:
        Dictionary mapping class label to RGBA tuple
    """
    base_names = ["blue", "red", "yellow", "green"]
    colors = {}

    for i, c in enumerate(classes[: len(base_names)]):
        colors[c] = mcolors.to_rgba(base_names[i])

    if len(classes) > len(base_names):
        extra_cmap = plt.cm.get_cmap("tab20", len(classes) - len(base_names))
        for j, c in enumerate(classes[len(base_names) :]):
            colors[c] = extra_cmap(j)

    return colors


def load_dunn_pvalues(stats_dir: Path | None) -> Dict[tuple, float]:
    """Load Dunn post-hoc p-values from stats results.

    Args:
        stats_dir: Directory containing Nonparametric_PostHoc.csv

    Returns:
        Dictionary mapping (feature, g1, g2) -> p_adj
        Groups are sorted alphabetically in the key.
    """
    if stats_dir is None:
        return {}

    path = Path(stats_dir) / "Nonparametric_PostHoc.csv"
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
        if not {"feature", "group1", "group2", "p_adj"}.issubset(df.columns):
            return {}

        lookup = {}
        for _, r in df.iterrows():
            g1, g2 = str(r["group1"]), str(r["group2"])
            key = (r["feature"],) + tuple(sorted([g1, g2]))
            lookup[key] = float(r["p_adj"]) if pd.notna(r["p_adj"]) else np.nan

        return lookup
    except Exception:
        return {}


def fallback_pairwise_pvalues(
    df: pd.DataFrame, features: List[str], label_col: str, g1: str, g2: str
) -> Dict[str, float]:
    """Compute fallback pairwise p-values using Mann-Whitney + BH-FDR.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        g1: First group
        g2: Second group

    Returns:
        Dictionary mapping feature -> adjusted p-value
    """
    raw_ps = []
    for feat in features:
        x = df.loc[df[label_col] == g1, feat].dropna().values
        y = df.loc[df[label_col] == g2, feat].dropna().values

        if len(x) == 0 or len(y) == 0:
            raw_ps.append(np.nan)
        else:
            try:
                _, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
                raw_ps.append(float(p))
            except Exception:
                raw_ps.append(np.nan)

    # BH-FDR correction
    mask = np.isfinite(raw_ps)
    pvals = np.array(raw_ps, dtype=float)
    p_adj = np.full_like(pvals, np.nan)

    if mask.sum() > 0:
        _, p_adj_subset, _, _ = multipletests(pvals[mask], method="fdr_bh")
        p_adj[mask] = p_adj_subset

    return {feat: p_adj[i] for i, feat in enumerate(features)}


def plot_boxplots(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    viz_dir: Path,
    config: VizConfig,
) -> Dict[str, Path]:
    """Create boxplots for all features.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        viz_dir: Visualization output directory
        config: VizConfig object

    Returns:
        Dictionary with paths to outputs
    """
    boxplot_dir = viz_dir / "boxplots"
    pdf_path = viz_dir / "boxplots_all.pdf"

    colors = choose_colors(classes)

    with PdfPages(pdf_path) as pdf:
        for feat in features:
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=config.fig_dpi)

            sns.boxplot(
                data=df,
                x=label_col,
                y=feat,
                order=classes,
                palette=[colors[c] for c in classes],
                fliersize=0,
                width=0.6,
                ax=ax,
            )

            sns.stripplot(
                data=df,
                x=label_col,
                y=feat,
                order=classes,
                color="black",
                size=np.sqrt(config.point_size),
                alpha=config.alpha_points * 0.6,
                ax=ax,
                jitter=0.2,
            )

            ax.set_title(f"Box & Whisker — {feat}")
            ax.set_xlabel(label_col)
            ax.set_ylabel(feat)

            for artist in ax.artists:
                artist.set_edgecolor("black")
                artist.set_linewidth(1.2)

            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()

            fig.savefig(boxplot_dir / f"{feat}.png")
            pdf.savefig(fig)
            plt.close(fig)

    return {"pdf": pdf_path, "png_dir": boxplot_dir}


def plot_foldchange_all_pairs(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    viz_dir: Path,
    config: VizConfig,
) -> List[Path]:
    """Create fold-change bar plots for all class pairs.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        viz_dir: Visualization output directory
        config: VizConfig object

    Returns:
        List of paths to created plots
    """
    fc_dir = viz_dir / "foldchange"
    colors = choose_colors(classes)
    pairs = list(itertools.combinations(classes, 2))
    outputs = []

    for g1, g2 in pairs:
        rows = []
        for feat in features:
            x = df.loc[df[label_col] == g1, feat].values
            y = df.loc[df[label_col] == g2, feat].values
            l2fc, v1, v2 = log2_fold_change(x, y, config.fc_center, config.fc_eps)

            rows.append(
                {
                    "feature": feat,
                    "group1": g1,
                    "group2": g2,
                    "center": config.fc_center,
                    "value1": v1,
                    "value2": v2,
                    "log2FC_g1_over_g2": l2fc,
                }
            )

        tbl = pd.DataFrame(rows).sort_values("log2FC_g1_over_g2", ascending=False)
        csv_path = fc_dir / f"foldchange_{g1}_vs_{g2}.csv"
        tbl.to_csv(csv_path, index=False)

        # Bar plot
        top_display = min(30, len(tbl))
        fig, ax = plt.subplots(figsize=(8, 0.33 * top_display + 1.5), dpi=config.fig_dpi)
        sub = tbl.head(top_display)

        ax.barh(sub["feature"], sub["log2FC_g1_over_g2"], color=colors[g1], edgecolor="black", linewidth=0.6)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel(f"log2FC ({g1}/{g2}) — center={config.fc_center}")
        ax.set_title(f"Top {top_display} Fold Changes: {g1} vs {g2}")
        fig.tight_layout()

        png_path = fc_dir / f"foldchange_{g1}_vs_{g2}.png"
        fig.savefig(png_path)
        plt.close(fig)

        outputs.extend([csv_path, png_path])

    return outputs


def plot_volcano_all_pairs(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    viz_dir: Path,
    config: VizConfig,
) -> List[Path]:
    """Create volcano plots for all class pairs.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        viz_dir: Visualization output directory
        config: VizConfig object

    Returns:
        List of paths to created plots
    """
    volcano_dir = viz_dir / "volcano"
    colors = choose_colors(classes)
    pairs = list(itertools.combinations(classes, 2))
    outputs = []

    # Load Dunn p-values if available
    dunn_lookup = load_dunn_pvalues(config.stats_dir)

    all_volcano_data = []

    for g1, g2 in pairs:
        records = []

        for feat in features:
            x = df.loc[df[label_col] == g1, feat].values
            y = df.loc[df[label_col] == g2, feat].values
            l2fc, v1, v2 = log2_fold_change(x, y, config.fc_center, config.fc_eps)

            # Try to get Dunn p_adj
            key = (feat,) + tuple(sorted([str(g1), str(g2)]))
            p_adj = dunn_lookup.get(key, np.nan)

            records.append(
                {
                    "feature": feat,
                    "group1": g1,
                    "group2": g2,
                    "center": config.fc_center,
                    "value1": v1,
                    "value2": v2,
                    "log2FC": l2fc,
                    "p_adj": p_adj,
                }
            )

        V = pd.DataFrame(records)

        # Fallback: compute MW + BH-FDR if no Dunn p-values
        if V["p_adj"].isna().all():
            fallback = fallback_pairwise_pvalues(df, features, label_col, g1, g2)
            V["p_adj"] = V["feature"].map(fallback)

        # Compute plotting values
        V["neglog10p"] = -np.log10(V["p_adj"].astype(float))
        sig_mask = (V["p_adj"] < config.alpha) & (V["log2FC"].abs() >= config.fc_thresh)

        # Save CSV
        csv_path = volcano_dir / f"volcano_{g1}_vs_{g2}.csv"
        V.sort_values("p_adj").to_csv(csv_path, index=False)

        # Volcano plot
        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=config.fig_dpi)

        # Non-significant
        ns = V[~sig_mask]
        ax.scatter(
            ns["log2FC"],
            ns["neglog10p"],
            s=config.point_size,
            c="lightgray",
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.4,
            label="NS",
        )

        # Significant up/down
        up = V[sig_mask & (V["log2FC"] > 0)]
        dn = V[sig_mask & (V["log2FC"] < 0)]

        ax.scatter(
            up["log2FC"],
            up["neglog10p"],
            s=config.point_size,
            c="#d62728",
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.6,
            label=f"{g1}↑",
        )

        ax.scatter(
            dn["log2FC"],
            dn["neglog10p"],
            s=config.point_size,
            c="#1f77b4",
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.6,
            label=f"{g2}↑",
        )

        # Thresholds
        ax.axvline(config.fc_thresh, color="gray", lw=1, ls="--")
        ax.axvline(-config.fc_thresh, color="gray", lw=1, ls="--")
        ax.axhline(-np.log10(config.alpha), color="gray", lw=1, ls="--")

        # Annotate top features
        V["score"] = V["neglog10p"] * V["log2FC"].abs()
        lab = V.sort_values("score", ascending=False).head(config.label_topk)

        for _, r in lab.iterrows():
            if np.isfinite(r["log2FC"]) and np.isfinite(r["neglog10p"]):
                ax.text(
                    r["log2FC"], r["neglog10p"], r["feature"], fontsize=8, ha="left", va="bottom"
                )

        ax.set_xlabel(f"log2FC ({g1}/{g2}) — center={config.fc_center}")
        ax.set_ylabel("-log10(p_adj)")
        ax.set_title(f"Volcano: {g1} vs {g2}")
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()

        png_path = volcano_dir / f"volcano_{g1}_vs_{g2}.png"
        fig.savefig(png_path)
        plt.close(fig)

        outputs.extend([csv_path, png_path])

        V["pair"] = f"{g1}_vs_{g2}"
        all_volcano_data.append(V)

    # Save combined volcano data
    if all_volcano_data:
        combined_csv = viz_dir / "volcano_all_pairs.csv"
        pd.concat(all_volcano_data, ignore_index=True).to_csv(combined_csv, index=False)
        outputs.append(combined_csv)

    return outputs


def plot_heatmap_top_features(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    viz_dir: Path,
    config: VizConfig,
) -> Path | None:
    """Create heatmap of top features.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        viz_dir: Visualization output directory
        config: VizConfig object

    Returns:
        Path to heatmap or None if skipped
    """
    if config.heatmap_topn <= 0:
        return None

    heatmap_dir = viz_dir / "heatmaps"
    colors = choose_colors(classes)

    # Load Dunn p-values to rank features
    dunn_lookup = load_dunn_pvalues(config.stats_dir)

    if dunn_lookup:
        # Rank by minimum p-value across pairs
        feat_pvals = {}
        for (feat, g1, g2), p in dunn_lookup.items():
            if feat not in feat_pvals:
                feat_pvals[feat] = []
            feat_pvals[feat].append(p)

        feat_min_p = {feat: min(ps) for feat, ps in feat_pvals.items()}
        ranked_features = sorted(feat_min_p.keys(), key=lambda f: feat_min_p[f])
    else:
        # No ranking available, use first N features
        ranked_features = features

    top = ranked_features[: min(config.heatmap_topn, len(ranked_features))]

    if len(top) == 0:
        return None

    # Z-score normalization
    X = df[top].copy()
    X = (X - X.mean()) / (X.std(ddof=1).replace(0, np.nan))

    # Order samples by label
    order = np.argsort(df[label_col].astype(str).values)
    X = X.iloc[order]

    plt.figure(figsize=(min(18, 0.35 * len(top) + 4), 0.02 * len(X) + 4), dpi=config.fig_dpi)
    sns.heatmap(X.T, cmap="vlag", center=0, cbar=True, xticklabels=False)
    plt.title("Top Features (z-scored)")
    plt.ylabel("Feature")
    plt.xlabel("Sample (ordered by class)")
    plt.tight_layout()

    out_path = heatmap_dir / "top_features_heatmap.png"
    plt.savefig(out_path)
    plt.close()

    return out_path
