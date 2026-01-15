"""Statistical tests (parametric and nonparametric)."""

from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Strictly require scikit-posthocs for Dunn (per user requirement)
try:
    import scikit_posthocs as sp

    HAS_SCPH = True
except ImportError:
    HAS_SCPH = False


def welch_ttest(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Perform Welch's t-test (unequal variances).

    Args:
        x: First group values
        y: Second group values

    Returns:
        Dictionary with keys: statistic, p_value, n1, n2, mean1, sd1, mean2, sd2
    """
    try:
        t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    except Exception:
        t_stat, p_val = np.nan, np.nan

    return {
        "statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        "n1": len(x),
        "n2": len(y),
        "mean1": float(np.mean(x)) if len(x) > 0 else np.nan,
        "sd1": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "mean2": float(np.mean(y)) if len(y) > 0 else np.nan,
        "sd2": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
    }


def anova_oneway(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Perform one-way ANOVA.

    Args:
        groups: List of group arrays

    Returns:
        Dictionary with keys: statistic, p_value, k_groups, N, df1, df2
    """
    try:
        F_stat, p_val = stats.f_oneway(*groups)
        N = sum(len(g) for g in groups)
        k = len(groups)
    except Exception:
        F_stat, p_val = np.nan, np.nan
        N = sum(len(g) for g in groups)
        k = len(groups)

    return {
        "statistic": float(F_stat) if np.isfinite(F_stat) else np.nan,
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        "k_groups": k,
        "N": N,
        "df1": k - 1,
        "df2": N - k,
    }


def tukey_posthoc(df: pd.DataFrame, feature: str, label_col: str, alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Perform Tukey HSD post-hoc test.

    Args:
        df: Dataframe with feature and label columns
        feature: Feature column name
        label_col: Label column name
        alpha: Significance level

    Returns:
        List of dictionaries, one per pairwise comparison
        Keys: feature, posthoc, group1, group2, mean_diff, p_adj, ci_low, ci_high, reject
    """
    y = df[feature].astype(float)
    g = df[label_col].astype(str)

    # Drop NaN
    valid = y.notna()
    y = y[valid]
    g = g[valid]

    try:
        tuk = pairwise_tukeyhsd(endog=y, groups=g, alpha=alpha)
        res = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])

        rows = []
        for _, r in res.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "posthoc": "Tukey HSD",
                    "group1": str(r["group1"]),
                    "group2": str(r["group2"]),
                    "mean_diff": float(r["meandiff"]),
                    "p_adj": float(r["p-adj"]),
                    "ci_low": float(r["lower"]),
                    "ci_high": float(r["upper"]),
                    "reject": bool(r["reject"]),
                }
            )
        return rows
    except Exception:
        return []


def kruskal_wallis(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Perform Kruskal-Wallis H test.

    Args:
        groups: List of group arrays

    Returns:
        Dictionary with keys: statistic, p_value, k_groups, df
    """
    try:
        H_stat, p_val = stats.kruskal(*groups)
        k = len(groups)
    except Exception:
        H_stat, p_val = np.nan, np.nan
        k = len(groups)

    return {
        "statistic": float(H_stat) if np.isfinite(H_stat) else np.nan,
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        "k_groups": k,
        "df": k - 1,
    }


def dunn_posthoc(
    df: pd.DataFrame, feature: str, label_col: str, p_adjust: str = "holm", alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Perform Dunn's post-hoc test with p-value adjustment.

    Args:
        df: Dataframe with feature and label columns
        feature: Feature column name
        label_col: Label column name
        p_adjust: P-value adjustment method
        alpha: Significance level (for reject flag)

    Returns:
        List of dictionaries, one per pairwise comparison
        Keys: feature, posthoc, group1, group2, p_adj, reject

    Raises:
        SystemExit: If scikit-posthocs is not installed
    """
    if not HAS_SCPH:
        raise SystemExit(
            "Dunn's post-hoc requires scikit-posthocs. "
            "Please install with: pip install scikit-posthocs"
        )

    try:
        ph = sp.posthoc_dunn(
            df[[feature, label_col]].dropna(), val_col=feature, group_col=label_col, p_adjust=p_adjust
        )

        # Melt to long format
        ph = ph.reset_index().rename(columns={"index": "group1"})
        ph = ph.melt(id_vars="group1", var_name="group2", value_name="p_adj")

        # Keep unique unordered pairs
        ph = ph[ph["group1"] < ph["group2"]]

        rows = []
        for _, r in ph.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "posthoc": f"Dunn ({p_adjust})",
                    "group1": str(r["group1"]),
                    "group2": str(r["group2"]),
                    "p_adj": float(r["p_adj"]),
                    "reject": bool(r["p_adj"] < alpha),
                }
            )
        return rows
    except SystemExit:
        raise
    except Exception:
        return []


def run_parametric_tests(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    normality_map: Dict[str, str],
    alpha: float = 0.05,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run parametric tests (Welch t-test or ANOVA + Tukey).

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        normality_map: Dict mapping feature -> normality flag
        alpha: Significance level

    Returns:
        Tuple of (overall_results, posthoc_results)
    """
    overall = []
    posthoc = []

    for feat in features:
        feature_normality = normality_map.get(feat, "Not tested")
        sub = df[[label_col, feat]].dropna()

        if sub.empty or sub[label_col].nunique() < 2:
            continue

        groups = [g[feat].values for _, g in sub.groupby(label_col)]
        group_labels = [str(name) for name, _ in sub.groupby(label_col)]
        k = len(groups)

        if k == 2:
            # Welch t-test
            result = welch_ttest(groups[0], groups[1])
            overall.append(
                {
                    "feature": feat,
                    "feature_normality": feature_normality,
                    "test": "Welch t-test",
                    "group1": group_labels[0],
                    "group2": group_labels[1],
                    **result,
                }
            )
        else:
            # ANOVA + Tukey
            result = anova_oneway(groups)
            overall.append(
                {
                    "feature": feat,
                    "feature_normality": feature_normality,
                    "test": "ANOVA",
                    **result,
                }
            )

            # Tukey post-hoc
            tukey_results = tukey_posthoc(sub, feat, label_col, alpha)
            for row in tukey_results:
                row["feature_normality"] = feature_normality
            posthoc.extend(tukey_results)

    return overall, posthoc


def run_nonparametric_tests(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    normality_map: Dict[str, str],
    dunn_adjust: str = "holm",
    alpha: float = 0.05,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run nonparametric tests (Kruskal-Wallis + Dunn).

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        normality_map: Dict mapping feature -> normality flag
        dunn_adjust: P-value adjustment method for Dunn
        alpha: Significance level

    Returns:
        Tuple of (overall_results, posthoc_results)
    """
    overall = []
    posthoc = []

    for feat in features:
        feature_normality = normality_map.get(feat, "Not tested")
        sub = df[[label_col, feat]].dropna()

        if sub.empty or sub[label_col].nunique() < 2:
            continue

        groups = [g[feat].values for _, g in sub.groupby(label_col)]
        k = len(groups)

        # Kruskal-Wallis (always, even for k=2)
        result = kruskal_wallis(groups)
        overall.append(
            {
                "feature": feat,
                "feature_normality": feature_normality,
                "test": "Kruskal–Wallis",
                **result,
            }
        )

        # Dunn post-hoc (always, even for k=2)
        dunn_results = dunn_posthoc(sub, feat, label_col, dunn_adjust, alpha)
        for row in dunn_results:
            row["feature_normality"] = feature_normality
        posthoc.extend(dunn_results)

    return overall, posthoc
