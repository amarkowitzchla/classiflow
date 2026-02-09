"""Tests for binary prediction column augmentation."""

import pandas as pd
import pytest

from classiflow.inference.predict import add_binary_prediction_columns


def test_add_binary_prediction_columns_with_probabilities():
    predictions = pd.DataFrame({
        "binary_task_score": [0.1, 0.8],
        "binary_task_pred": [0, 1],
    })
    labels = pd.Series(["WNT_alpha", "WNT_beta"])

    augmented = add_binary_prediction_columns(
        predictions.copy(),
        labels=labels,
        positive_class="WNT_beta",
    )

    assert "predicted_label" in augmented.columns
    assert augmented["predicted_label"].tolist() == ["WNT_alpha", "WNT_beta"]
    assert "predicted_proba_WNT_alpha" in augmented.columns
    assert "predicted_proba_WNT_beta" in augmented.columns
    assert augmented["predicted_proba_WNT_beta"].tolist() == pytest.approx([0.1, 0.8])
    assert augmented["predicted_proba_WNT_alpha"].tolist() == pytest.approx([0.9, 0.2])
    assert augmented["predicted_proba"].tolist() == pytest.approx([0.9, 0.8])


def test_add_binary_prediction_columns_skips_nonprobability_scores():
    predictions = pd.DataFrame({
        "binary_task_score": [-1.5, 0.7],
        "binary_task_pred": [0, 1],
    })
    labels = pd.Series(["neg", "pos"])

    augmented = add_binary_prediction_columns(predictions.copy(), labels=labels, positive_class="pos")

    assert "predicted_label" in augmented.columns
    assert "predicted_proba_neg" not in augmented.columns
    assert "predicted_proba_pos" not in augmented.columns
