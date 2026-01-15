"""Tests for stats preprocessing."""

import pytest
import pandas as pd
import numpy as np

from classiflow.stats.preprocess import (
    select_numeric_features,
    prepare_data,
    compute_class_stats,
)


def test_select_numeric_features():
    """Test numeric feature selection."""
    df = pd.DataFrame(
        {
            "label": ["A", "B", "A", "B"],
            "feat1": [1.0, 2.0, 3.0, 4.0],
            "feat2": [5.0, 6.0, 7.0, 8.0],
            "feat3": ["x", "y", "z", "w"],
        }
    )

    features = select_numeric_features(df, "label")
    assert set(features) == {"feat1", "feat2"}


def test_select_numeric_features_with_whitelist():
    """Test numeric feature selection with whitelist."""
    df = pd.DataFrame(
        {
            "label": ["A", "B"],
            "feat1": [1.0, 2.0],
            "feat2": [3.0, 4.0],
            "feat3": [5.0, 6.0],
        }
    )

    features = select_numeric_features(df, "label", whitelist=["feat1", "feat3"])
    assert set(features) == {"feat1", "feat3"}


def test_select_numeric_features_with_blacklist():
    """Test numeric feature selection with blacklist."""
    df = pd.DataFrame(
        {
            "label": ["A", "B"],
            "feat1": [1.0, 2.0],
            "feat2": [3.0, 4.0],
            "feat3": [5.0, 6.0],
        }
    )

    features = select_numeric_features(df, "label", blacklist=["feat2"])
    assert set(features) == {"feat1", "feat3"}


def test_prepare_data_basic():
    """Test basic data preparation."""
    df = pd.DataFrame(
        {
            "label": ["A", "B", "A", "B", "C"],
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    df_prep, features, classes = prepare_data(df, "label")

    assert len(df_prep) == 5
    assert set(features) == {"feat1", "feat2"}
    assert len(classes) == 3


def test_prepare_data_with_class_subset():
    """Test data preparation with class subset."""
    df = pd.DataFrame(
        {
            "label": ["A", "B", "A", "B", "C"],
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    df_prep, features, classes = prepare_data(df, "label", classes=["A", "B"])

    assert len(df_prep) == 4
    assert classes == ["A", "B"]


def test_prepare_data_drops_missing_labels():
    """Test that rows with missing labels are dropped."""
    df = pd.DataFrame(
        {
            "label": ["A", None, "B", "A"],
            "feat1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    df_prep, features, classes = prepare_data(df, "label")

    assert len(df_prep) == 3


def test_prepare_data_raises_on_missing_label_col():
    """Test that ValueError is raised if label column missing."""
    df = pd.DataFrame({"feat1": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Label column 'label' not found"):
        prepare_data(df, "label")


def test_prepare_data_raises_on_insufficient_classes():
    """Test that ValueError is raised if < 2 classes."""
    df = pd.DataFrame(
        {
            "label": ["A", "A", "A"],
            "feat1": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="At least 2 classes required"):
        prepare_data(df, "label")


def test_compute_class_stats():
    """Test computing descriptive stats by class."""
    df = pd.DataFrame(
        {
            "label": ["A", "A", "A", "B", "B", "B"],
            "feat1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )

    stats = compute_class_stats(df, "feat1", "label", ["A", "B"])

    assert len(stats) == 2
    assert stats.loc[stats["class"] == "A", "n"].values[0] == 3
    assert stats.loc[stats["class"] == "A", "mean"].values[0] == pytest.approx(2.0)
    assert stats.loc[stats["class"] == "B", "mean"].values[0] == pytest.approx(20.0)
