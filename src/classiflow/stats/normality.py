"""Normality testing with Shapiro-Wilk."""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def shapiro_safe(x: np.ndarray) -> Tuple[float, float, int]:
    """Perform Shapiro-Wilk test with safe handling.

    Args:
        x: Array of values

    Returns:
        Tuple of (W_statistic, p_value, n_valid)

    Notes:
        - Returns (nan, nan, n) if n < 3 or constant values
        - Subsamples to 5000 if n > 5000 (SciPy accuracy recommendation)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    # Need at least 3 observations
    if n < 3:
        return np.nan, np.nan, n

    # Check for constant values
    if np.allclose(np.nanstd(x), 0.0):
        return np.nan, np.nan, n

    # Subsample if too large (SciPy recommendation)
    if n > 5000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=5000, replace=False)
        n = 5000

    W, p = stats.shapiro(x)
    return float(W), float(p), int(n)


def check_normality_by_class(
    df: pd.DataFrame, feature: str, label_col: str, classes: List[str], min_n: int = 3
) -> pd.DataFrame:
    """Test normality within each class for a feature.

    Args:
        df: Input dataframe
        feature: Feature column name
        label_col: Label column name
        classes: List of class labels
        min_n: Minimum n required to perform test

    Returns:
        DataFrame with columns: feature, class, n, test, W, p_value
    """
    rows = []
    for c in classes:
        x = df.loc[df[label_col] == c, feature].dropna()
        W, p, n = shapiro_safe(x.values)

        rows.append(
            {"feature": feature, "class": c, "n": n, "test": "Shapiro–Wilk", "W": W, "p_value": p}
        )

    return pd.DataFrame(rows)


def determine_normality_flag(
    normality_by_class: pd.DataFrame, alpha: float, min_n: int
) -> str:
    """Determine overall normality flag for a feature.

    Args:
        normality_by_class: Output from test_normality_by_class()
        alpha: Significance level
        min_n: Minimum n required for testing

    Returns:
        One of: "Normal", "Not normal", "Not tested"

    Logic:
        - "Not tested": No class had n >= min_n
        - "Normal": ALL classes with n >= min_n have p >= alpha
        - "Not normal": Otherwise
    """
    # Filter to classes with sufficient n
    valid = normality_by_class[normality_by_class["n"] >= min_n]

    if len(valid) == 0:
        return "Not tested"

    # Check if all valid classes pass normality
    all_pass = (valid["p_value"] >= alpha).all()

    return "Normal" if all_pass else "Not normal"


def check_normality_all_features(
    df: pd.DataFrame, features: List[str], label_col: str, classes: List[str], alpha: float, min_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test normality for all features across all classes.

    Args:
        df: Input dataframe
        features: List of feature names
        label_col: Label column name
        classes: List of class labels
        alpha: Significance level
        min_n: Minimum n required for testing

    Returns:
        Tuple of (normality_summary, normality_detail)

        normality_summary columns: feature, test_name, p_value, normality
        normality_detail columns: feature, class, n, test, W, p_value
    """
    summary_rows = []
    detail_rows = []

    for feat in features:
        # Test per class
        detail = check_normality_by_class(df, feat, label_col, classes, min_n)
        detail_rows.append(detail)

        # Determine overall flag
        normality_flag = determine_normality_flag(detail, alpha, min_n)

        # Get minimum p-value for summary
        valid_p = detail.loc[detail["n"] >= min_n, "p_value"]
        p_min = float(np.nanmin(valid_p)) if len(valid_p) > 0 else np.nan

        summary_rows.append(
            {
                "feature": feat,
                "test_name": "Shapiro–Wilk",
                "p_value": p_min,
                "normality": normality_flag,
            }
        )

    normality_summary = pd.DataFrame(
        summary_rows, columns=["feature", "test_name", "p_value", "normality"]
    )
    normality_detail = pd.concat(detail_rows, ignore_index=True)

    return normality_summary, normality_detail
