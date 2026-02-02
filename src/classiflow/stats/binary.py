"""Binary (2-class) statistical analysis utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from classiflow.stats.effects import hedges_g, rank_biserial, log2_fold_change


def _normality_lookup(
    normality_by_class: pd.DataFrame,
) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """Build lookup for per-feature/per-class normality p-values and n."""
    lookup: Dict[Tuple[str, str], Tuple[float, int]] = {}
    if normality_by_class.empty:
        return lookup

    for _, row in normality_by_class.iterrows():
        feature = str(row["feature"])
        cls = str(row["class"])
        p_val = float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan
        n_val = int(row["n"]) if pd.notna(row["n"]) else 0
        lookup[(feature, cls)] = (p_val, n_val)

    return lookup


def _binary_normality_flag(
    p_a: float, p_b: float, n_a: int, n_b: int, alpha: float, min_n: int
) -> str:
    """Determine binary normality flag based on both classes."""
    if n_a < min_n or n_b < min_n:
        return "Not tested"
    if np.isfinite(p_a) and np.isfinite(p_b) and p_a >= alpha and p_b >= alpha:
        return "Normal"
    return "Not normal"


def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """Adjust p-values with statsmodels multipletests."""
    p_adj = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.any():
        _, adj, _, _ = multipletests(pvals[mask], method=method)
        p_adj[mask] = adj
    return p_adj


def binary_feature_tests(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    classes: List[str],
    normality_by_class: pd.DataFrame,
    alpha: float,
    min_n: int,
    p_adjust: str,
    fc_center: str = "median",
    fc_eps: float = 1e-9,
) -> pd.DataFrame:
    """Run per-feature binary tests with normality-based dispatch."""
    class_a, class_b = classes[0], classes[1]
    normality_lookup = _normality_lookup(normality_by_class)
    rows = []

    for feat in features:
        x = df.loc[df[label_col] == class_a, feat].dropna().astype(float).values
        y = df.loc[df[label_col] == class_b, feat].dropna().astype(float).values

        n_a, n_b = len(x), len(y)
        mean_a = float(np.mean(x)) if n_a > 0 else np.nan
        mean_b = float(np.mean(y)) if n_b > 0 else np.nan
        sd_a = float(np.std(x, ddof=1)) if n_a > 1 else np.nan
        sd_b = float(np.std(y, ddof=1)) if n_b > 1 else np.nan
        median_a = float(np.median(x)) if n_a > 0 else np.nan
        median_b = float(np.median(y)) if n_b > 0 else np.nan

        norm_p_a, norm_n_a = normality_lookup.get((feat, class_a), (np.nan, 0))
        norm_p_b, norm_n_b = normality_lookup.get((feat, class_b), (np.nan, 0))

        normality_p_a = norm_p_a if norm_n_a >= min_n else np.nan
        normality_p_b = norm_p_b if norm_n_b >= min_n else np.nan

        normality_flag = _binary_normality_flag(
            normality_p_a, normality_p_b, n_a, n_b, alpha, min_n
        )

        is_parametric = normality_flag == "Normal"
        test_type = "ttest_welch" if is_parametric else "mannwhitney"

        statistic = np.nan
        p_value = np.nan
        if n_a > 0 and n_b > 0:
            try:
                if is_parametric:
                    statistic, p_value = sp_stats.ttest_ind(x, y, equal_var=False)
                else:
                    statistic, p_value = sp_stats.mannwhitneyu(
                        x, y, alternative="two-sided"
                    )
            except Exception:
                statistic, p_value = np.nan, np.nan

        log2fc, fc_center_a, fc_center_b = log2_fold_change(
            x, y, center=fc_center, eps=fc_eps
        )

        rows.append(
            {
                "feature": feat,
                "group1": class_a,
                "group2": class_b,
                "n1": n_a,
                "n2": n_b,
                "mean1": mean_a,
                "mean2": mean_b,
                "sd1": sd_a,
                "sd2": sd_b,
                "median1": median_a,
                "median2": median_b,
                "normality_p1": normality_p_a,
                "normality_p2": normality_p_b,
                "normality": normality_flag,
                "test_type": test_type,
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "log2fc": log2fc,
                "fc_center1": fc_center_a,
                "fc_center2": fc_center_b,
                "hedges_g": hedges_g(x, y) if is_parametric else np.nan,
                "rank_biserial": rank_biserial(x, y) if not is_parametric else np.nan,
                "delta_mean": (
                    mean_a - mean_b
                    if np.isfinite(mean_a) and np.isfinite(mean_b)
                    else np.nan
                ),
                "delta_median": (
                    median_a - median_b
                    if np.isfinite(median_a) and np.isfinite(median_b)
                    else np.nan
                ),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    p_adj = _adjust_pvalues(results["p_value"].to_numpy(dtype=float), method=p_adjust)
    results["p_adj"] = p_adj
    results["reject"] = results["p_adj"] < alpha
    results.loc[~np.isfinite(results["p_adj"]), "reject"] = False

    return results


def split_binary_test_tables(
    binary_results: pd.DataFrame,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Split binary results into parametric/nonparametric tables."""
    if binary_results.empty:
        return [], [], []

    param_rows = []
    nonparam_rows = []
    posthoc_rows = []

    for _, row in binary_results.iterrows():
        base = {
            "feature": row["feature"],
            "feature_normality": row["normality"],
            "group1": row["group1"],
            "group2": row["group2"],
            "n1": int(row["n1"]),
            "n2": int(row["n2"]),
            "mean1": row["mean1"],
            "sd1": row["sd1"],
            "mean2": row["mean2"],
            "sd2": row["sd2"],
            "statistic": row["statistic"],
            "p_value": row["p_value"],
        }

        posthoc_rows.append(
            {
                "feature": row["feature"],
                "feature_normality": row["normality"],
                "posthoc": "Binary (adj)",
                "group1": row["group1"],
                "group2": row["group2"],
                "p_adj": row["p_adj"],
                "reject": bool(row["reject"]),
            }
        )

        if row["test_type"] == "ttest_welch":
            param_rows.append({**base, "test": "Welch t-test"})
        else:
            nonparam_rows.append({**base, "test": "Mann–Whitney U"})

    return param_rows, nonparam_rows, posthoc_rows


def build_binary_pairwise_summary(binary_results: pd.DataFrame) -> pd.DataFrame:
    """Return binary results as pairwise summary table."""
    if binary_results.empty:
        return binary_results
    return binary_results.copy()
