"""Tests for the calibration metrics helpers."""

import numpy as np
import pandas as pd

from classiflow.metrics.calibration import compute_probability_quality


def test_compute_probability_quality_binary_perfect():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        dtype=float,
    )

    metrics, curve = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["0", "1"],
        bins=5,
    )

    assert "brier" in metrics and "ece" in metrics
    assert isinstance(curve, pd.DataFrame)


def test_compute_probability_quality_multiclass_perfect():
    y_true = ["a", "b", "c"]
    y_pred = ["a", "b", "c"]
    y_proba = np.eye(3)

    metrics, curve = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["a", "b", "c"],
        bins=3,
    )

    assert "ece" in metrics
    assert isinstance(curve, pd.DataFrame)
