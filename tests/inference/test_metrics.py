"""Tests for inference metric edge cases."""

import numpy as np
import pandas as pd

from classiflow.inference.api import _build_true_labels, _compute_metrics
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


def test_build_true_labels_combines_hierarchical_levels():
    metadata = pd.DataFrame(
        {
            "label_l1": ["A", "B"],
            "label_l2": ["A1", "B2"],
        }
    )

    y_true = _build_true_labels(
        metadata,
        run_type="hierarchical",
        label_col="label_l1",
        label_col_secondary="label_l2",
    )

    assert y_true.tolist() == ["A::A1", "B::B2"]


def test_compute_metrics_prefers_y_true_for_hierarchical_runs():
    predictions = pd.DataFrame(
        {
            "label_l1": ["A", "B"],
            "y_true": ["A::A1", "B::B2"],
            "predicted_label": ["A::A1", "B::B2"],
            "predicted_label_L1": ["A", "B"],
            "predicted_proba_A::A1": [0.95, 0.05],
            "predicted_proba_B::B2": [0.05, 0.95],
            "predicted_proba_L1_A": [0.95, 0.05],
            "predicted_proba_L1_B": [0.05, 0.95],
        }
    )

    metrics, _ = _compute_metrics(
        predictions,
        label_col="label_l1",
        run_type="hierarchical",
    )

    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["hierarchical"]["L1"]["accuracy"] == 1.0


def test_compute_metrics_ignores_l1_probabilities_for_overall_hierarchical_metrics():
    predictions = pd.DataFrame(
        {
            "label_l1": ["A", "B"],
            "y_true": ["A::A1", "B::B2"],
            "predicted_label": ["A::A1", "B::B2"],
            "predicted_label_L1": ["A", "B"],
            "predicted_proba_L1_A": [0.95, 0.05],
            "predicted_proba_L1_B": [0.05, 0.95],
        }
    )

    metrics, _ = _compute_metrics(
        predictions,
        label_col="label_l1",
        run_type="hierarchical",
    )

    assert metrics["overall"]["accuracy"] == 1.0
    assert "probability_quality" not in metrics["overall"]
