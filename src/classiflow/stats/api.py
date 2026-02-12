"""Public API for statistical analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from classiflow.stats.config import StatsConfig, VizConfig
from classiflow.stats.preprocess import prepare_data
from classiflow.stats.normality import check_normality_all_features
from classiflow.stats.tests import run_parametric_tests, run_nonparametric_tests
from classiflow.stats.binary import (
    binary_feature_tests,
    split_binary_test_tables,
    build_binary_pairwise_summary,
)
from classiflow.stats import reports
from classiflow.stats.excel import (
    write_publication_workbook,
    write_legacy_workbook,
    write_legacy_csvs,
)


def run_stats(
    data_csv: Path | str,
    label_col: str,
    outdir: Path | str,
    classes: Optional[List[str]] = None,
    alpha: float = 0.05,
    min_n: int = 3,
    dunn_adjust: str = "holm",
    feature_whitelist: Optional[List[str]] = None,
    feature_blacklist: Optional[List[str]] = None,
    top_n_features: int = 30,
    write_legacy_csv: bool = True,
    write_legacy_xlsx: bool = True,
    fc_center: str = "median",
    fc_eps: float = 1e-9,
) -> Dict[str, Any]:
    """Run complete statistical analysis pipeline.

    Args:
        data_csv: Path to input CSV with features and labels
        label_col: Name of the label/class column
        outdir: Output directory for results
        classes: Optional subset/order of classes to include
        alpha: Significance threshold (default: 0.05)
        min_n: Minimum n per class to run Shapiro-Wilk (default: 3)
        dunn_adjust: P-value adjustment method for Dunn test (default: "holm")
        feature_whitelist: Optional list of features to include
        feature_blacklist: Optional list of features to exclude
        top_n_features: Number of top features to include in summary (default: 30)
        write_legacy_csv: Whether to write legacy CSV outputs (default: True)
        write_legacy_xlsx: Whether to write legacy stats_results.xlsx (default: True)
        fc_center: Center measure for fold change ("mean" or "median")
        fc_eps: Pseudocount for fold change ratios

    Returns:
        Dictionary with results and output paths

    Example:
        >>> from classiflow.stats import run_stats
        >>> results = run_stats(
        ...     data_csv="data.csv",
        ...     label_col="diagnosis",
        ...     outdir="derived/stats_results",
        ...     alpha=0.05
        ... )
        >>> print(f"Results saved to: {results['publication_xlsx']}")
    """
    config = StatsConfig(
        data_csv=Path(data_csv),
        label_col=label_col,
        outdir=Path(outdir),
        classes=classes,
        alpha=alpha,
        min_n=min_n,
        dunn_adjust=dunn_adjust,
        feature_whitelist=feature_whitelist,
        feature_blacklist=feature_blacklist,
        top_n_features=top_n_features,
        write_legacy_csv=write_legacy_csv,
        write_legacy_xlsx=write_legacy_xlsx,
    )

    return run_stats_from_config(config, fc_center=fc_center, fc_eps=fc_eps)


def run_stats_from_config(
    config: StatsConfig, fc_center: str = "median", fc_eps: float = 1e-9
) -> Dict[str, Any]:
    """Run statistical analysis from a StatsConfig object.

    Args:
        config: StatsConfig object
        fc_center: Center measure for fold change
        fc_eps: Pseudocount for fold change

    Returns:
        Dictionary with results and output paths

    Notes:
        - Binary (2-class): Welch's t-test when both classes pass normality,
          otherwise Mann–Whitney U with per-feature p-value adjustment.
        - Multiclass (3+): ANOVA/Tukey or Kruskal–Wallis/Dunn as configured.
    """
    from classiflow.data import load_table

    print(f"Loading data from {config.data_csv}...")
    df = load_table(config.data_csv)

    # Prepare data
    print("Preparing data...")
    df, features, classes = prepare_data(
        df,
        config.label_col,
        config.classes,
        config.feature_whitelist,
        config.feature_blacklist,
    )

    print(f"  • Classes: {len(classes)} → {classes}")
    print(f"  • Features: {len(features)}")
    print(f"  • Samples: {len(df)}")

    # Class counts
    class_counts = df[config.label_col].value_counts().to_dict()

    # Create output directory
    stats_dir = config.outdir / "stats_results"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 1. Normality testing
    # ──────────────────────────────────────────────────────────────
    print("\n[1/5] Running normality tests (Shapiro–Wilk)...")
    normality_summary, normality_by_class = check_normality_all_features(
        df, features, config.label_col, classes, config.alpha, config.min_n
    )

    # Build normality map for later use
    normality_map = normality_summary.set_index("feature")["normality"].to_dict()

    # ──────────────────────────────────────────────────────────────
    # 2-3. Statistical tests (dispatch by class count)
    # ──────────────────────────────────────────────────────────────
    if len(classes) < 2:
        raise ValueError("At least 2 classes required for statistical analysis")

    if len(classes) == 2:
        print("[2/5] Running binary tests (Welch t-test or Mann–Whitney U)...")
        binary_results = binary_feature_tests(
            df,
            features,
            config.label_col,
            classes,
            normality_by_class,
            config.alpha,
            config.min_n,
            config.dunn_adjust,
            fc_center=fc_center,
            fc_eps=fc_eps,
        )

        print("[3/5] Preparing binary summaries...")
        param_overall, nonparam_overall, nonparam_posthoc = split_binary_test_tables(
            binary_results
        )
        param_posthoc = []

        parametric_overall = reports.format_parametric_overall(param_overall)
        parametric_posthoc = reports.format_parametric_posthoc(param_posthoc)
        nonparametric_overall = reports.format_nonparametric_overall(nonparam_overall)
        nonparametric_posthoc = reports.format_nonparametric_posthoc(nonparam_posthoc)
        pairwise_summary = build_binary_pairwise_summary(binary_results)
    else:
        # ──────────────────────────────────────────────────────────────
        # 2. Parametric tests
        # ──────────────────────────────────────────────────────────────
        print("[2/5] Running parametric tests (Welch t-test / ANOVA + Tukey)...")
        param_overall, param_posthoc = run_parametric_tests(
            df, features, config.label_col, normality_map, config.alpha
        )

        parametric_overall = reports.format_parametric_overall(param_overall)
        parametric_posthoc = reports.format_parametric_posthoc(param_posthoc)

        # ──────────────────────────────────────────────────────────────
        # 3. Nonparametric tests
        # ──────────────────────────────────────────────────────────────
        print("[3/5] Running nonparametric tests (Kruskal–Wallis + Dunn)...")
        nonparam_overall, nonparam_posthoc = run_nonparametric_tests(
            df, features, config.label_col, normality_map, config.dunn_adjust, config.alpha
        )

        nonparametric_overall = reports.format_nonparametric_overall(nonparam_overall)
        nonparametric_posthoc = reports.format_nonparametric_posthoc(nonparam_posthoc)

        # ──────────────────────────────────────────────────────────────
        # 4. Build publication tables
        # ──────────────────────────────────────────────────────────────
        print("[4/5] Building publication-ready tables...")

        # Run manifest
        run_manifest = reports.build_run_manifest(
            config.data_csv,
            config.label_col,
            classes,
            class_counts,
            config.alpha,
            config.min_n,
            config.dunn_adjust,
            config.top_n_features,
        )

        # Descriptives by class
        descriptives_by_class = reports.build_descriptives_by_class(
            df, features, config.label_col, classes
        )

        # Pairwise summary with effect sizes
        pairwise_summary = reports.build_pairwise_summary(
            df,
            features,
            config.label_col,
            classes,
            nonparametric_posthoc,
            normality_map,
            fc_center,
            fc_eps,
            config.alpha,
        )

        # Top features
        top_features_per_pair, top_features_overall = reports.build_top_features(
            pairwise_summary, config.top_n_features
        )

    if len(classes) == 2:
        print("[4/5] Building publication-ready tables...")
        run_manifest = reports.build_run_manifest(
            config.data_csv,
            config.label_col,
            classes,
            class_counts,
            config.alpha,
            config.min_n,
            config.dunn_adjust,
            config.top_n_features,
        )
        descriptives_by_class = reports.build_descriptives_by_class(
            df, features, config.label_col, classes
        )
        top_features_per_pair, top_features_overall = reports.build_top_features(
            pairwise_summary, config.top_n_features
        )

    # ──────────────────────────────────────────────────────────────
    # 5. Write outputs
    # ──────────────────────────────────────────────────────────────
    print("[5/5] Writing outputs...")

    # Publication workbook
    pub_xlsx = write_publication_workbook(
        stats_dir,
        run_manifest,
        descriptives_by_class,
        normality_summary,
        normality_by_class,
        parametric_overall,
        parametric_posthoc,
        nonparametric_overall,
        nonparametric_posthoc,
        pairwise_summary,
        top_features_per_pair,
        top_features_overall,
    )
    print(f"  • {pub_xlsx}")

    # Legacy workbook
    legacy_xlsx = None
    if config.write_legacy_xlsx:
        legacy_xlsx = write_legacy_workbook(
            stats_dir,
            normality_summary,
            normality_by_class,
            parametric_overall,
            parametric_posthoc,
            nonparametric_overall,
            nonparametric_posthoc,
        )
        print(f"  • {legacy_xlsx}")

    # Legacy CSVs
    if config.write_legacy_csv:
        write_legacy_csvs(
            stats_dir,
            normality_summary,
            normality_by_class,
            parametric_overall,
            parametric_posthoc,
            nonparametric_overall,
            nonparametric_posthoc,
        )
        print(f"  • Legacy CSVs in {stats_dir}")

    print("\n✓ Statistical analysis complete.")

    return {
        "publication_xlsx": pub_xlsx,
        "legacy_xlsx": legacy_xlsx,
        "stats_dir": stats_dir,
        "n_features": len(features),
        "n_classes": len(classes),
        "n_samples": len(df),
        "classes": classes,
        "features": features,
        "pairwise_summary": pairwise_summary,
        "top_features_overall": top_features_overall,
    }


def run_visualizations(
    data_csv: Path | str,
    label_col: str,
    outdir: Path | str,
    stats_dir: Optional[Path | str] = None,
    classes: Optional[List[str]] = None,
    alpha: float = 0.05,
    fc_thresh: float = 1.0,
    fc_center: str = "median",
    fc_eps: float = 1e-9,
    label_topk: int = 12,
    boxplot_ncols: int = 3,
    heatmap_topn: int = 30,
    fig_dpi: int = 160,
    point_size: float = 48.0,
    alpha_points: float = 0.9,
) -> Dict[str, Any]:
    """Run statistical visualizations.

    Args:
        data_csv: Path to input CSV
        label_col: Name of label column
        outdir: Output directory for visualizations
        stats_dir: Optional directory with stats results for volcano plots
        classes: Optional subset/order of classes
        alpha: Significance threshold for volcano plots
        fc_thresh: |log2FC| threshold for volcano plots
        fc_center: Center measure for fold change
        fc_eps: Pseudocount for fold change
        label_topk: Number of top features to annotate on volcano
        boxplot_ncols: Number of columns for boxplot grid
        heatmap_topn: Number of top features for heatmap (0 = skip)
        fig_dpi: Figure DPI
        point_size: Scatter point size
        alpha_points: Scatter point transparency

    Returns:
        Dictionary with output paths

    Example:
        >>> from classiflow.stats import run_visualizations
        >>> results = run_visualizations(
        ...     data_csv="data.csv",
        ...     label_col="diagnosis",
        ...     outdir="derived/viz",
        ...     stats_dir="derived/stats_results"
        ... )
    """
    config = VizConfig(
        data_csv=Path(data_csv),
        label_col=label_col,
        outdir=Path(outdir),
        stats_dir=Path(stats_dir) if stats_dir else None,
        classes=classes,
        alpha=alpha,
        fc_thresh=fc_thresh,
        fc_center=fc_center,
        fc_eps=fc_eps,
        label_topk=label_topk,
        boxplot_ncols=boxplot_ncols,
        heatmap_topn=heatmap_topn,
        fig_dpi=fig_dpi,
        point_size=point_size,
        alpha_points=alpha_points,
    )

    return run_visualizations_from_config(config)


def run_visualizations_from_config(config: VizConfig) -> Dict[str, Any]:
    """Run visualizations from a VizConfig object.

    Args:
        config: VizConfig object

    Returns:
        Dictionary with output paths
    """
    # Import here to avoid heavy dependencies if only running stats
    from classiflow.stats.viz import (
        plot_boxplots,
        plot_foldchange_all_pairs,
        plot_volcano_all_pairs,
        plot_heatmap_top_features,
    )
    from classiflow.stats.clustering import plot_all_projections

    from classiflow.data import load_table

    print(f"Loading data from {config.data_csv}...")
    df = load_table(config.data_csv)

    from classiflow.stats.preprocess import prepare_data

    df, features, classes = prepare_data(df, config.label_col, config.classes)

    print(f"  • Classes: {len(classes)}")
    print(f"  • Features: {len(features)}")

    # Create output directories
    viz_dir = config.outdir / "viz"
    (viz_dir / "boxplots").mkdir(parents=True, exist_ok=True)
    (viz_dir / "foldchange").mkdir(parents=True, exist_ok=True)
    (viz_dir / "volcano").mkdir(parents=True, exist_ok=True)
    (viz_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    (viz_dir / "projections").mkdir(parents=True, exist_ok=True)

    # Boxplots
    print("\n[1/5] Creating boxplots...")
    boxplot_outputs = plot_boxplots(df, features, config.label_col, classes, viz_dir, config)

    # Fold-change plots
    print("[2/5] Creating fold-change plots...")
    foldchange_outputs = plot_foldchange_all_pairs(
        df, features, config.label_col, classes, viz_dir, config
    )

    # Volcano plots
    print("[3/5] Creating volcano plots...")
    volcano_outputs = plot_volcano_all_pairs(
        df, features, config.label_col, classes, viz_dir, config
    )

    # Heatmap
    heatmap_path = None
    if config.heatmap_topn > 0:
        print("[4/5] Creating heatmap...")
        heatmap_path = plot_heatmap_top_features(
            df, features, config.label_col, classes, viz_dir, config
        )

    # Dimensionality reduction plots (UMAP, t-SNE, LDA)
    print("[5/5] Creating dimensionality reduction plots...")
    projection_outputs = plot_all_projections(
        df, features, config.label_col, classes, viz_dir / "projections", config
    )

    print("\n✓ Visualizations complete.")

    return {
        "viz_dir": viz_dir,
        "boxplots": boxplot_outputs,
        "foldchange": foldchange_outputs,
        "volcano": volcano_outputs,
        "heatmap": heatmap_path,
        "projections": projection_outputs,
    }
