"""Tests for metrics computation."""

import pytest
import numpy as np

from classiflow.metrics import compute_binary_metrics


def test_compute_binary_metrics_perfect():
    """Test metrics with perfect predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    metrics = compute_binary_metrics(y_true, scores)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_binary_metrics_random():
    """Test metrics with random predictions."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100)
    scores = np.random.rand(100)

    metrics = compute_binary_metrics(y_true, scores)

    # Should have all required keys
    required_keys = ["n", "pos_rate", "accuracy", "balanced_accuracy",
                     "precision", "recall", "f1", "mcc", "roc_auc"]
    for key in required_keys:
        assert key in metrics

    # Values should be in valid range
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_compute_binary_metrics_imbalanced():
    """Test metrics with imbalanced data."""
    y_true = np.array([0] * 90 + [1] * 10)
    scores = np.array([0.1] * 90 + [0.9] * 10)

    metrics = compute_binary_metrics(y_true, scores)

    assert metrics["n"] == 100
    assert metrics["pos_rate"] == 0.1
    assert metrics["accuracy"] > 0.9
