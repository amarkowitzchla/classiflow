"""Tests for normality testing."""

import pytest
import pandas as pd
import numpy as np

from classiflow.stats.normality import (
    shapiro_safe,
    check_normality_by_class,
    determine_normality_flag,
    check_normality_all_features,
)


def test_shapiro_safe_normal_data():
    """Test Shapiro-Wilk on normally distributed data."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 50)

    W, p, n = shapiro_safe(x)

    assert n == 50
    assert 0 <= W <= 1
    assert 0 <= p <= 1


def test_shapiro_safe_insufficient_data():
    """Test Shapiro-Wilk with insufficient data."""
    x = np.array([1.0, 2.0])  # n < 3

    W, p, n = shapiro_safe(x)

    assert n == 2
    assert np.isnan(W)
    assert np.isnan(p)


def test_shapiro_safe_constant_data():
    """Test Shapiro-Wilk with constant data."""
    x = np.array([5.0, 5.0, 5.0, 5.0])

    W, p, n = shapiro_safe(x)

    assert n == 4
    assert np.isnan(W)
    assert np.isnan(p)


def test_shapiro_safe_handles_nan():
    """Test Shapiro-Wilk handles NaN values."""
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])

    W, p, n = shapiro_safe(x)

    assert n == 5  # NaN excluded


def test_shapiro_safe_large_sample():
    """Test Shapiro-Wilk subsamples large datasets."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 10000)

    W, p, n = shapiro_safe(x)

    # Should subsample to 5000
    assert n == 5000
    assert 0 <= W <= 1


def test_check_normality_by_class():
    """Test normality testing per class."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50,
            "feat1": list(np.random.normal(0, 1, 50)) + list(np.random.normal(10, 1, 50)),
        }
    )

    result = check_normality_by_class(df, "feat1", "label", ["A", "B"])

    assert len(result) == 2
    assert set(result["class"]) == {"A", "B"}
    assert result["test"].iloc[0] == "Shapiro–Wilk"


def test_determine_normality_flag_normal():
    """Test normality flag determination for normal data."""
    normality_df = pd.DataFrame(
        {
            "feature": ["feat1", "feat1"],
            "class": ["A", "B"],
            "n": [50, 50],
            "p_value": [0.5, 0.7],
        }
    )

    flag = determine_normality_flag(normality_df, alpha=0.05, min_n=3)

    assert flag == "Normal"


def test_determine_normality_flag_not_normal():
    """Test normality flag determination for non-normal data."""
    normality_df = pd.DataFrame(
        {
            "feature": ["feat1", "feat1"],
            "class": ["A", "B"],
            "n": [50, 50],
            "p_value": [0.5, 0.01],  # One class fails
        }
    )

    flag = determine_normality_flag(normality_df, alpha=0.05, min_n=3)

    assert flag == "Not normal"


def test_determine_normality_flag_not_tested():
    """Test normality flag when no class has sufficient n."""
    normality_df = pd.DataFrame(
        {
            "feature": ["feat1", "feat1"],
            "class": ["A", "B"],
            "n": [2, 2],  # Both below min_n
            "p_value": [np.nan, np.nan],
        }
    )

    flag = determine_normality_flag(normality_df, alpha=0.05, min_n=3)

    assert flag == "Not tested"


def test_check_normality_all_features():
    """Test normality testing for all features."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20,
            "feat1": list(np.random.normal(0, 1, 20)) + list(np.random.normal(5, 1, 20)),
            "feat2": list(np.random.normal(10, 2, 20)) + list(np.random.normal(15, 2, 20)),
        }
    )

    summary, detail = check_normality_all_features(
        df, ["feat1", "feat2"], "label", ["A", "B"], alpha=0.05, min_n=3
    )

    assert len(summary) == 2  # Two features
    assert len(detail) == 4  # Two features × two classes
    assert set(summary["feature"]) == {"feat1", "feat2"}
    assert all(summary["normality"].isin(["Normal", "Not normal", "Not tested"]))
