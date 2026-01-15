"""Tests for feature drift detection."""

import pytest
import numpy as np
import pandas as pd

from classiflow.validation.drift import (
    compute_feature_summary,
    compute_drift_scores,
    detect_drift,
)


def test_compute_feature_summary():
    """Test computing feature summary statistics."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": [1.1, 2.2, np.nan, 4.4, 5.5],
    })

    summary = compute_feature_summary(df)

    assert "feature1" in summary
    assert "feature2" in summary
    assert "feature3" in summary

    # Check feature1 stats
    assert summary["feature1"]["mean"] == pytest.approx(3.0)
    assert summary["feature1"]["std"] > 0
    assert summary["feature1"]["missing_rate"] == 0.0

    # Check feature3 with missing value
    assert summary["feature3"]["missing_rate"] > 0


def test_compute_drift_scores_no_drift():
    """Test drift scores when there's no drift."""
    train_summary = {
        "feature1": {
            "mean": 10.0,
            "std": 2.0,
            "median": 10.0,
            "q25": 8.0,
            "q75": 12.0,
            "missing_rate": 0.0,
        },
    }

    # Inference summary identical to training
    inf_summary = train_summary.copy()

    drift_df = compute_drift_scores(train_summary, inf_summary)

    assert len(drift_df) == 1
    assert drift_df["z_shift"].iloc[0] == pytest.approx(0.0)
    assert drift_df["missing_delta"].iloc[0] == pytest.approx(0.0)


def test_compute_drift_scores_mean_shift():
    """Test drift scores with mean shift."""
    train_summary = {
        "feature1": {
            "mean": 10.0,
            "std": 2.0,
            "median": 10.0,
            "q25": 8.0,
            "q75": 12.0,
            "missing_rate": 0.0,
        },
    }

    inf_summary = {
        "feature1": {
            "mean": 16.0,  # 3 std above training mean
            "std": 2.0,
            "median": 16.0,
            "q25": 14.0,
            "q75": 18.0,
            "missing_rate": 0.0,
        },
    }

    drift_df = compute_drift_scores(train_summary, inf_summary)

    # z_shift = (16 - 10) / 2 = 3.0
    assert drift_df["z_shift"].iloc[0] == pytest.approx(3.0)


def test_compute_drift_scores_missing_rate_change():
    """Test drift scores with missing rate change."""
    train_summary = {
        "feature1": {
            "mean": 10.0,
            "std": 2.0,
            "median": 10.0,
            "q25": 8.0,
            "q75": 12.0,
            "missing_rate": 0.0,
        },
    }

    inf_summary = {
        "feature1": {
            "mean": 10.0,
            "std": 2.0,
            "median": 10.0,
            "q25": 8.0,
            "q75": 12.0,
            "missing_rate": 0.2,  # 20% missing
        },
    }

    drift_df = compute_drift_scores(train_summary, inf_summary)

    # missing_delta = 0.2 - 0.0 = 0.2
    assert drift_df["missing_delta"].iloc[0] == pytest.approx(0.2)


def test_detect_drift_no_flags():
    """Test drift detection with no flagged features."""
    drift_df = pd.DataFrame({
        "feature": ["feat1", "feat2"],
        "z_shift": [0.5, 1.0],
        "missing_delta": [0.01, 0.02],
        "median_shift": [0.3, 0.4],
        "abs_z_shift": [0.5, 1.0],
        "abs_missing_delta": [0.01, 0.02],
        "abs_median_shift": [0.3, 0.4],
    })

    flagged, warnings = detect_drift(drift_df)

    assert len(flagged) == 0
    assert len(warnings) == 0


def test_detect_drift_with_flags():
    """Test drift detection with flagged features."""
    drift_df = pd.DataFrame({
        "feature": ["feat1", "feat2", "feat3"],
        "z_shift": [5.0, 1.0, 0.5],  # feat1 exceeds threshold
        "missing_delta": [0.01, 0.15, 0.02],  # feat2 exceeds threshold
        "median_shift": [0.3, 0.4, 0.5],
        "abs_z_shift": [5.0, 1.0, 0.5],
        "abs_missing_delta": [0.01, 0.15, 0.02],
        "abs_median_shift": [0.3, 0.4, 0.5],
    })

    flagged, warnings = detect_drift(
        drift_df,
        z_threshold=3.0,
        missing_threshold=0.1,
        median_threshold=2.0,
    )

    assert len(flagged) == 2  # feat1 and feat2
    assert len(warnings) > 0


def test_feature_summary_with_all_missing():
    """Test feature summary with all missing values."""
    df = pd.DataFrame({
        "feature1": [np.nan, np.nan, np.nan],
    })

    summary = compute_feature_summary(df)

    assert summary["feature1"]["missing_rate"] == 1.0
    assert np.isnan(summary["feature1"]["mean"])
