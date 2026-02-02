"""Build publication-ready statistical reports and tables."""

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import itertools

import pandas as pd
import numpy as np

from classiflow import __version__
from classiflow.stats.preprocess import compute_class_stats
from classiflow.stats.effects import compute_all_effect_sizes


def build_run_manifest(
    data_csv: Path,
    label_col: str,
    classes: List[str],
    class_counts: Dict[str, int],
    alpha: float,
    min_n: int,
    dunn_adjust: str,
    top_n_features: int,
) -> pd.DataFrame:
    """Build run manifest sheet.

    Returns:
        DataFrame with metadata about the analysis run
    """
    rows = [
        {"parameter": "dataset", "value": str(data_csv.name)},
        {"parameter": "label_column", "value": label_col},
        {"parameter": "timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"parameter": "alpha", "value": alpha},
        {"parameter": "min_n", "value": min_n},
        {"parameter": "dunn_p_adjust", "value": dunn_adjust},
        {"parameter": "top_n_features", "value": top_n_features},
        {"parameter": "package_version", "value": __version__},
        {"parameter": "n_classes", "value": len(classes)},
        {"parameter": "classes_order", "value": ", ".join(classes)},
    ]

    # Add class counts
    for cls, cnt in class_counts.items():
        rows.append({"parameter": f"n_{cls}", "value": cnt})

    return pd.DataFrame(rows, columns=["parameter", "value"])


def build_descriptives_by_class(
    df: pd.DataFrame, features: List[str], label_col: str, classes: List[str]
) -> pd.DataFrame:
    """Build descriptive statistics by class.

    Returns:
        DataFrame with columns: feature, class, n, n_missing, mean, sd, median, q25, q75, iqr
    """
    all_stats = []

    for feat in features:
        stats_df = compute_class_stats(df, feat, label_col, classes)
        stats_df.insert(0, "feature", feat)
        all_stats.append(stats_df)

    return pd.concat(all_stats, ignore_index=True)


def build_pairwise_summary(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    dunn_posthoc: pd.DataFrame,
    normality_map: Dict[str, str],
    fc_center: str = "median",
    fc_eps: float = 1e-9,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build pairwise summary table with effect sizes and p-values.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        dunn_posthoc: Dunn post-hoc results (with p_adj)
        normality_map: Dict mapping feature -> normality flag
        fc_center: Center measure for fold change
        fc_eps: Pseudocount for fold change
        alpha: Significance threshold

    Returns:
        DataFrame with columns:
            feature, group1, group2, normality, log2fc, fc_center1, fc_center2,
            cohen_d, cliff_delta, rank_biserial, p_adj, reject
    """
    # Build lookup for Dunn p-values
    dunn_lookup = {}
    for _, row in dunn_posthoc.iterrows():
        key = (row["feature"], tuple(sorted([str(row["group1"]), str(row["group2"])])))
        dunn_lookup[key] = (float(row["p_adj"]), bool(row["reject"]))

    rows = []
    pairs = list(itertools.combinations(classes, 2))

    for feat in features:
        normality = normality_map.get(feat, "Not tested")

        for g1, g2 in pairs:
            # Get data
            x = df.loc[df[label_col] == g1, feat].dropna().values
            y = df.loc[df[label_col] == g2, feat].dropna().values

            # Effect sizes
            effects = compute_all_effect_sizes(x, y, fc_center, fc_eps)

            # Get p_adj from Dunn
            key = (feat, tuple(sorted([str(g1), str(g2)])))
            p_adj, reject = dunn_lookup.get(key, (np.nan, False))

            rows.append(
                {
                    "feature": feat,
                    "group1": g1,
                    "group2": g2,
                    "normality": normality,
                    "log2fc": effects["log2fc"],
                    "fc_center1": effects["fc_center_x"],
                    "fc_center2": effects["fc_center_y"],
                    "cohen_d": effects["cohen_d"],
                    "cliff_delta": effects["cliff_delta"],
                    "rank_biserial": effects["rank_biserial"],
                    "p_adj": p_adj,
                    "reject": reject,
                }
            )

    return pd.DataFrame(rows)


def build_top_features(
    pairwise_summary: pd.DataFrame, top_n: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top features summaries.

    Args:
        pairwise_summary: Output from build_pairwise_summary()
        top_n: Number of top features to extract

    Returns:
        Tuple of (top_per_pair, overall_ranking)

        top_per_pair: Top N features per pair sorted by p_adj
        overall_ranking: Features ranked by minimum p_adj across all pairs
    """
    # Top per pair
    top_per_pair = (
        pairwise_summary.sort_values("p_adj")
        .groupby(["group1", "group2"])
        .head(top_n)
        .reset_index(drop=True)
    )

    # Overall ranking: take minimum p_adj per feature across all pairs
    overall = (
        pairwise_summary.groupby("feature")
        .agg({"p_adj": "min", "reject": "any"})
        .reset_index()
        .sort_values("p_adj")
        .head(top_n)
        .reset_index(drop=True)
    )
    overall.columns = ["feature", "min_p_adj", "significant_any_pair"]

    return top_per_pair, overall


def format_parametric_overall(overall: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format parametric overall results into clean DataFrame.

    Handles both Welch t-test (k=2) and ANOVA (k>=3) results.
    """
    if not overall:
        return pd.DataFrame()

    df = pd.DataFrame(overall)

    # Reorder columns for clarity
    if "group1" in df.columns:  # Welch t-test
        col_order = [
            "feature",
            "feature_normality",
            "test",
            "group1",
            "group2",
            "n1",
            "n2",
            "mean1",
            "sd1",
            "mean2",
            "sd2",
            "statistic",
            "p_value",
        ]
    else:  # ANOVA
        col_order = [
            "feature",
            "feature_normality",
            "test",
            "k_groups",
            "N",
            "df1",
            "df2",
            "statistic",
            "p_value",
        ]

    existing_cols = [c for c in col_order if c in df.columns]
    return df[existing_cols]


def format_parametric_posthoc(posthoc: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format parametric post-hoc results (Tukey) into clean DataFrame."""
    if not posthoc:
        return pd.DataFrame()

    df = pd.DataFrame(posthoc)

    col_order = [
        "feature",
        "feature_normality",
        "posthoc",
        "group1",
        "group2",
        "mean_diff",
        "p_adj",
        "ci_low",
        "ci_high",
        "reject",
    ]

    existing_cols = [c for c in col_order if c in df.columns]
    return df[existing_cols]


def format_nonparametric_overall(overall: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format nonparametric overall results (Kruskal-Wallis) into clean DataFrame."""
    if not overall:
        return pd.DataFrame()

    df = pd.DataFrame(overall)

    if "group1" in df.columns:  # Mann-Whitney U
        col_order = [
            "feature",
            "feature_normality",
            "test",
            "group1",
            "group2",
            "n1",
            "n2",
            "statistic",
            "p_value",
        ]
    else:  # Kruskal-Wallis
        col_order = [
            "feature",
            "feature_normality",
            "test",
            "k_groups",
            "df",
            "statistic",
            "p_value",
        ]

    existing_cols = [c for c in col_order if c in df.columns]
    return df[existing_cols]


def format_nonparametric_posthoc(posthoc: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format nonparametric post-hoc results (Dunn) into clean DataFrame."""
    if not posthoc:
        return pd.DataFrame()

    df = pd.DataFrame(posthoc)

    col_order = ["feature", "feature_normality", "posthoc", "group1", "group2", "p_adj", "reject"]

    existing_cols = [c for c in col_order if c in df.columns]
    return df[existing_cols]
