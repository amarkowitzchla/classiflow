"""Dimensionality reduction and clustering visualizations (UMAP, t-SNE, LDA)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from classiflow.stats.config import VizConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")


def choose_colors(classes: List[str]) -> Dict[str, tuple]:
    """Map classes to colors."""
    base_names = ["blue", "red", "yellow", "green"]
    colors = {}

    for i, c in enumerate(classes[: len(base_names)]):
        colors[c] = mcolors.to_rgba(base_names[i])

    if len(classes) > len(base_names):
        extra_cmap = plt.cm.get_cmap("tab20", len(classes) - len(base_names))
        for j, c in enumerate(classes[len(base_names) :]):
            colors[c] = extra_cmap(j)

    return colors


def plot_umap(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    outdir: Path,
    config: VizConfig,
    supervised: bool = False,
) -> Optional[Path]:
    """Create UMAP projection plot.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        outdir: Output directory
        config: VizConfig object
        supervised: Use supervised UMAP (default: False)

    Returns:
        Path to saved plot or None if UMAP not available
    """
    try:
        from umap import UMAP
    except ImportError:
        print("  ⚠ UMAP not available (install with: pip install umap-learn)")
        return None

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print(f"    • Running UMAP (supervised={supervised})...")

    # Prepare data
    X = df[features].values
    y_str = df[label_col].astype(str).values

    # Impute missing values
    X = SimpleImputer(strategy="median").fit_transform(X)

    # Standardize
    X = StandardScaler().fit_transform(X)

    # Encode labels for supervised UMAP
    y_encoded = None
    if supervised:
        class_to_int = {c: i for i, c in enumerate(classes)}
        y_encoded = np.array([class_to_int[s] for s in y_str], dtype=np.int64)

    # Run UMAP
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        random_state=42,
        verbose=False,
    )

    if supervised:
        emb = reducer.fit_transform(X, y=y_encoded)
    else:
        emb = reducer.fit_transform(X)

    # Save embedding
    emb_df = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"])
    emb_df[label_col] = y_str
    csv_path = outdir / "umap_embedding.csv"
    emb_df.to_csv(csv_path, index=False)

    # Plot
    colors = choose_colors(classes)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=config.fig_dpi)

    for c in classes:
        mask = emb_df[label_col] == c
        ax.scatter(
            emb_df.loc[mask, "UMAP1"],
            emb_df.loc[mask, "UMAP2"],
            s=config.point_size,
            c=[colors[c]],
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.8,
            label=c,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    mode = "Supervised" if supervised else "Unsupervised"
    ax.set_title(f"UMAP Projection ({mode})", fontsize=14, fontweight="bold")

    if len(classes) <= 15:
        ax.legend(loc="best", frameon=True, fontsize=9)
    else:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title=label_col,
        )
        fig.subplots_adjust(right=0.78)

    ax.grid(False)
    fig.tight_layout()

    png_path = outdir / "umap_projection.png"
    svg_path = outdir / "umap_projection.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"      ✓ Saved: {png_path}")
    return png_path


def plot_tsne(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    outdir: Path,
    config: VizConfig,
    perplexity: int = 30,
) -> Optional[Path]:
    """Create t-SNE projection plot.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        outdir: Output directory
        config: VizConfig object
        perplexity: t-SNE perplexity parameter (default: 30)

    Returns:
        Path to saved plot or None if error
    """
    from sklearn.manifold import TSNE
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print(f"    • Running t-SNE (perplexity={perplexity})...")

    # Prepare data
    X = df[features].values
    y_str = df[label_col].astype(str).values

    # Impute missing values
    X = SimpleImputer(strategy="median").fit_transform(X)

    # Standardize
    X = StandardScaler().fit_transform(X)

    # Adjust perplexity if needed
    n_samples = len(X)
    perplexity = min(perplexity, (n_samples - 1) // 3)

    if perplexity < 5:
        print(f"      ⚠ Insufficient samples for t-SNE (n={n_samples}), skipping")
        return None

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        verbose=0,
    )

    emb = tsne.fit_transform(X)

    # Save embedding
    emb_df = pd.DataFrame(emb, columns=["tSNE1", "tSNE2"])
    emb_df[label_col] = y_str
    csv_path = outdir / "tsne_embedding.csv"
    emb_df.to_csv(csv_path, index=False)

    # Plot
    colors = choose_colors(classes)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=config.fig_dpi)

    for c in classes:
        mask = emb_df[label_col] == c
        ax.scatter(
            emb_df.loc[mask, "tSNE1"],
            emb_df.loc[mask, "tSNE2"],
            s=config.point_size,
            c=[colors[c]],
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.8,
            label=c,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE Projection (perplexity={perplexity})", fontsize=14, fontweight="bold")

    if len(classes) <= 15:
        ax.legend(loc="best", frameon=True, fontsize=9)
    else:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title=label_col,
        )
        fig.subplots_adjust(right=0.78)

    ax.grid(False)
    fig.tight_layout()

    png_path = outdir / "tsne_projection.png"
    svg_path = outdir / "tsne_projection.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"      ✓ Saved: {png_path}")
    return png_path


def plot_lda(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    outdir: Path,
    config: VizConfig,
) -> Optional[Path]:
    """Create Linear Discriminant Analysis projection plot.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        outdir: Path
        config: VizConfig object

    Returns:
        Path to saved plot or None if error
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print("    • Running LDA...")

    # Prepare data
    X = df[features].values
    y_str = df[label_col].astype(str).values

    # Impute missing values
    X = SimpleImputer(strategy="median").fit_transform(X)

    # Standardize
    X = StandardScaler().fit_transform(X)

    # Encode labels
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_int[s] for s in y_str])

    # LDA can project to at most (n_classes - 1) dimensions
    n_components = min(2, len(classes) - 1)

    if n_components < 1:
        print("      ⚠ LDA requires at least 2 classes, skipping")
        return None

    # Run LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    try:
        emb = lda.fit_transform(X, y)
    except Exception as e:
        print(f"      ⚠ LDA failed: {e}")
        return None

    # Save embedding
    if n_components == 1:
        emb_df = pd.DataFrame(emb, columns=["LD1"])
        emb_df["LD2"] = 0  # Add dummy second dimension for 2-class case
    else:
        emb_df = pd.DataFrame(emb, columns=["LD1", "LD2"])

    emb_df[label_col] = y_str
    csv_path = outdir / "lda_embedding.csv"
    emb_df.to_csv(csv_path, index=False)

    # Plot
    colors = choose_colors(classes)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=config.fig_dpi)

    for c in classes:
        mask = emb_df[label_col] == c
        ax.scatter(
            emb_df.loc[mask, "LD1"],
            emb_df.loc[mask, "LD2"],
            s=config.point_size,
            c=[colors[c]],
            alpha=config.alpha_points,
            edgecolors="black",
            linewidths=0.8,
            label=c,
        )

    ax.set_xlabel("LD 1", fontsize=12)
    ax.set_ylabel("LD 2" if n_components == 2 else "LD 2 (dummy)", fontsize=12)

    # Add variance explained
    if hasattr(lda, "explained_variance_ratio_"):
        var_ratio = lda.explained_variance_ratio_
        if n_components == 1:
            title = f"LDA Projection (LD1: {var_ratio[0]:.1%} variance)"
        else:
            title = f"LDA Projection (LD1: {var_ratio[0]:.1%}, LD2: {var_ratio[1]:.1%})"
    else:
        title = "LDA Projection"

    ax.set_title(title, fontsize=14, fontweight="bold")

    if len(classes) <= 15:
        ax.legend(loc="best", frameon=True, fontsize=9)
    else:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title=label_col,
        )
        fig.subplots_adjust(right=0.78)

    ax.grid(False)
    fig.tight_layout()

    png_path = outdir / "lda_projection.png"
    svg_path = outdir / "lda_projection.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"      ✓ Saved: {png_path}")
    return png_path


def plot_all_projections(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    outdir: Path,
    config: VizConfig,
    methods: Optional[List[str]] = None,
) -> Dict[str, Optional[Path]]:
    """Create all dimensionality reduction plots.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        outdir: Output directory
        config: VizConfig object
        methods: List of methods to run (default: ["umap", "tsne", "lda"])

    Returns:
        Dictionary mapping method name to output path
    """
    if methods is None:
        methods = ["umap", "tsne", "lda"]

    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    if "umap" in methods:
        results["umap"] = plot_umap(df, features, label_col, classes, outdir, config, supervised=False)

    if "tsne" in methods:
        results["tsne"] = plot_tsne(df, features, label_col, classes, outdir, config, perplexity=30)

    if "lda" in methods:
        results["lda"] = plot_lda(df, features, label_col, classes, outdir, config)

    return results
