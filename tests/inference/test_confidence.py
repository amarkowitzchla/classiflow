"""Tests for confidence metrics."""

import pytest
import numpy as np
import pandas as pd

from classiflow.inference.confidence import (
    compute_confidence_metrics,
    assign_confidence_buckets,
    annotate_predictions_with_confidence,
)


def test_compute_confidence_metrics_multiclass():
    """Test confidence metrics for multiclass predictions."""
    # Mock probabilities (n_samples=5, n_classes=3)
    probabilities = np.array([
        [0.7, 0.2, 0.1],  # High confidence
        [0.5, 0.3, 0.2],  # Medium confidence
        [0.4, 0.35, 0.25],  # Low confidence
        [0.9, 0.05, 0.05],  # Very high confidence
        [0.33, 0.33, 0.34],  # Very low confidence
    ])

    confidence_df = compute_confidence_metrics(probabilities)

    # Check columns
    assert "confidence_max_proba" in confidence_df.columns
    assert "confidence_margin" in confidence_df.columns
    assert "confidence_entropy" in confidence_df.columns

    # Check values
    assert confidence_df["confidence_max_proba"].iloc[0] == pytest.approx(0.7)
    assert confidence_df["confidence_margin"].iloc[0] == pytest.approx(0.5)  # 0.7 - 0.2

    # Entropy should be higher for more uncertain predictions
    entropy_uncertain = confidence_df["confidence_entropy"].iloc[4]
    entropy_certain = confidence_df["confidence_entropy"].iloc[3]
    assert entropy_uncertain > entropy_certain


def test_compute_confidence_metrics_binary():
    """Test confidence metrics for binary predictions."""
    # Binary case: single probability column
    probabilities = np.array([0.8, 0.6, 0.55, 0.9, 0.45])

    confidence_df = compute_confidence_metrics(probabilities)

    assert len(confidence_df) == 5
    assert "confidence_max_proba" in confidence_df.columns


def test_assign_confidence_buckets():
    """Test confidence bucket assignment."""
    confidence_scores = pd.Series([0.95, 0.85, 0.65, 0.45, 0.92])

    buckets = assign_confidence_buckets(confidence_scores)

    assert len(buckets) == 5
    assert buckets.iloc[0] == "high"  # 0.95
    assert buckets.iloc[1] == "medium"  # 0.85
    assert buckets.iloc[2] == "low"  # 0.65


def test_assign_confidence_buckets_custom_thresholds():
    """Test confidence buckets with custom thresholds."""
    confidence_scores = pd.Series([0.95, 0.85, 0.65, 0.45])

    thresholds = {"high_min": 0.8, "medium_min": 0.6, "low_min": 0.0}
    buckets = assign_confidence_buckets(confidence_scores, thresholds=thresholds)

    assert buckets.iloc[0] == "high"  # 0.95
    assert buckets.iloc[1] == "high"  # 0.85
    assert buckets.iloc[2] == "medium"  # 0.65
    assert buckets.iloc[3] == "low"  # 0.45


def test_annotate_predictions_with_confidence():
    """Test annotating predictions DataFrame with confidence."""
    predictions_df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "predicted_label": ["A", "B", "A"],
    })

    probabilities = np.array([
        [0.8, 0.2],
        [0.6, 0.4],
        [0.55, 0.45],
    ])

    annotated_df = annotate_predictions_with_confidence(
        predictions_df,
        probabilities,
    )

    # Check original columns preserved
    assert "sample_id" in annotated_df.columns
    assert "predicted_label" in annotated_df.columns

    # Check confidence columns added
    assert "confidence_max_proba" in annotated_df.columns
    assert "confidence_margin" in annotated_df.columns
    assert "confidence_entropy" in annotated_df.columns
    assert "confidence_bucket" in annotated_df.columns


def test_confidence_metrics_shape_consistency():
    """Test that confidence metrics have correct shape."""
    n_samples = 100
    n_classes = 5

    probabilities = np.random.dirichlet(np.ones(n_classes), size=n_samples)

    confidence_df = compute_confidence_metrics(probabilities)

    assert len(confidence_df) == n_samples
    assert len(confidence_df.columns) >= 3
