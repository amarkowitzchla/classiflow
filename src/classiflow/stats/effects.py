"""Effect size calculations for pairwise comparisons."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import stats as sp_stats


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size.

    Args:
        x: First group values
        y: Second group values

    Returns:
        Cohen's d (pooled standard deviation)

    Notes:
        Returns NaN if insufficient data or zero variance
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)

    if nx < 2 or ny < 2:
        return np.nan

    # Sample variances
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled variance
    sp2 = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2) if (nx + ny - 2) > 0 else np.nan

    if not np.isfinite(sp2) or sp2 <= 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / np.sqrt(sp2)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Hedges' g effect size (small-sample corrected Cohen's d).

    Args:
        x: First group values
        y: Second group values

    Returns:
        Hedges' g

    Notes:
        Returns NaN if insufficient data or zero variance
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)

    d = cohen_d(x, y)
    if not np.isfinite(d):
        return np.nan

    df = nx + ny - 2
    if df <= 0:
        return np.nan

    correction = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * correction


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's Delta effect size.

    Args:
        x: First group values
        y: Second group values

    Returns:
        Cliff's Delta in range [-1, 1]

    Notes:
        - Delta = (# pairs where x > y - # pairs where x < y) / (nx * ny)
        - Returns NaN if either group is empty
        - Interpretation: |delta| < 0.147: negligible, < 0.33: small,
          < 0.474: medium, >= 0.474: large
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)

    if nx == 0 or ny == 0:
        return np.nan

    # Count dominances
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])

    return (greater - less) / (nx * ny)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate rank-biserial correlation from Mann-Whitney U.

    Args:
        x: First group values
        y: Second group values

    Returns:
        Rank-biserial correlation in range [-1, 1]

    Notes:
        Computed as: r = 1 - (2*U) / (nx * ny)
        Returns NaN if insufficient data
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)

    if nx == 0 or ny == 0:
        return np.nan

    try:
        U, _ = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        return 1.0 - (2.0 * U) / (nx * ny)
    except Exception:
        return np.nan


def log2_fold_change(
    x: np.ndarray, y: np.ndarray, center: str = "median", eps: float = 1e-9
) -> Tuple[float, float, float]:
    """Calculate log2 fold change.

    Args:
        x: First group values (numerator)
        y: Second group values (denominator)
        center: "mean" or "median"
        eps: Pseudocount to avoid division by zero

    Returns:
        Tuple of (log2FC, center_x, center_y)

    Notes:
        log2FC = log2((center(x) + eps) / (center(y) + eps))
        Returns (NaN, NaN, NaN) if insufficient data
    """
    x, y = np.asarray(x), np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan

    if center == "mean":
        v1 = float(np.nanmean(x))
        v2 = float(np.nanmean(y))
    else:  # median
        v1 = float(np.nanmedian(x))
        v2 = float(np.nanmedian(y))

    if not np.isfinite(v1) or not np.isfinite(v2):
        return np.nan, v1, v2

    log2fc = float(np.log2((v1 + eps) / (v2 + eps)))

    return log2fc, v1, v2


def compute_all_effect_sizes(
    x: np.ndarray, y: np.ndarray, fc_center: str = "median", fc_eps: float = 1e-9
) -> dict:
    """Compute all effect sizes for a pairwise comparison.

    Args:
        x: First group values
        y: Second group values
        fc_center: Center measure for fold change ("mean" or "median")
        fc_eps: Pseudocount for fold change

    Returns:
        Dictionary with keys:
            - cohen_d: Cohen's d
            - cliff_delta: Cliff's delta
            - rank_biserial: Rank-biserial correlation
            - log2fc: Log2 fold change (x over y)
            - fc_center_x: Center value for x
            - fc_center_y: Center value for y
    """
    x_clean = x[np.isfinite(x)]
    y_clean = y[np.isfinite(y)]

    log2fc, center_x, center_y = log2_fold_change(x_clean, y_clean, fc_center, fc_eps)

    return {
        "cohen_d": cohen_d(x_clean, y_clean),
        "cliff_delta": cliff_delta(x_clean, y_clean),
        "rank_biserial": rank_biserial(x_clean, y_clean),
        "log2fc": log2fc,
        "fc_center_x": center_x,
        "fc_center_y": center_y,
    }
