"""Tests for inference metric edge cases."""

import numpy as np

from classiflow.inference.metrics import compute_classification_metrics
from classiflow.inference.plots import generate_all_plots


def test_classification_metrics_warn_when_model_classes_miss_test_labels():
    y_true = np.array(["A", "A"])
    y_pred = np.array(["B", "B"])
    y_proba = np.array([[0.8, 0.2], [0.7, 0.3]])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_proba=y_proba,
        class_names=["B", "C"],
    )

    assert "warnings" in metrics
    assert metrics["confusion_matrix"]["labels"] == ["B", "C", "A"]
    assert metrics["confusion_matrix"]["matrix"][2][0] == 2


def test_inference_plots_warn_and_continue_with_missing_test_labels(tmp_path):
    y_true = np.array(["A", "A"])
    y_pred = np.array(["B", "B"])
    y_proba = np.array([[0.8, 0.2], [0.7, 0.3]])

    plot_paths = generate_all_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["B", "C"],
        output_dir=tmp_path,
    )

    assert "confusion_matrix" in plot_paths
    assert plot_paths["confusion_matrix"].exists()


def test_probability_metrics_use_evaluable_subset_for_partial_class_overlap():
    y_true = np.array(["A", "C"])
    y_pred = np.array(["A", "B"])
    y_proba = np.array([[0.8, 0.2], [0.7, 0.3]])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_proba=y_proba,
        class_names=["A", "B"],
    )

    assert "roc_auc" in metrics
    assert metrics["confusion_matrix"]["labels"] == ["A", "B", "C"]
    assert any("n_excluded=1" in warning for warning in metrics["warnings"])
