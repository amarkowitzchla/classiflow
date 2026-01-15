"""Tests for AdaptiveSMOTE."""

import pytest
import numpy as np
import pandas as pd

from classiflow.models import AdaptiveSMOTE


def test_adaptive_smote_normal():
    """Test SMOTE with sufficient samples."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.array([0] * 80 + [1] * 20)

    sampler = AdaptiveSMOTE(k_max=5, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    # Should have resampled
    assert len(y_res) > len(y)
    # Majority and minority should be balanced
    assert np.sum(y_res == 0) == np.sum(y_res == 1)


def test_adaptive_smote_too_small():
    """Test SMOTE with too few minority samples."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])  # Only 1 minority sample

    sampler = AdaptiveSMOTE(k_max=5, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    # Should pass through unchanged
    assert len(y_res) == len(y)
    np.testing.assert_array_equal(X_res, X)
    np.testing.assert_array_equal(y_res, y)


def test_adaptive_smote_non_binary():
    """Test SMOTE with non-binary labels."""
    X = np.random.randn(50, 10)
    y = np.array([0, 1, 2] * 16 + [0, 1])  # Multiclass

    sampler = AdaptiveSMOTE(k_max=5, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    # Should resample minority classes to match the majority class size
    assert len(y_res) == 51
    counts = pd.Series(y_res).value_counts().to_dict()
    assert counts[0] == counts[1] == counts[2]


def test_adaptive_smote_k_adaptation():
    """Test k_neighbors adaptation."""
    X = np.random.randn(30, 10)
    y = np.array([0] * 25 + [1] * 5)  # 5 minority samples

    sampler = AdaptiveSMOTE(k_max=10, random_state=42)  # k_max > minority
    X_res, y_res = sampler.fit_resample(X, y)

    # Should adapt k to minority-1 = 4
    # Should successfully resample
    assert len(y_res) > len(y)
