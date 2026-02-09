"""Tests for calibration metrics across binary/multiclass/meta/hierarchical modes."""

from __future__ import annotations

import numpy as np

from classiflow.metrics.calibration import compute_probability_quality


def test_probability_quality_binary_known_values():
    y_true = ["0", "1", "0", "1"]
    y_pred = ["0", "1", "0", "1"]
    y_proba = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
        ],
        dtype=float,
    )

    metrics, curves = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["0", "1"],
        bins=4,
        mode="binary",
    )

    y_pos = np.array([0.2, 0.9, 0.3, 0.8])
    y_bin = np.array([0.0, 1.0, 0.0, 1.0])
    expected_brier_binary = float(np.mean((y_pos - y_bin) ** 2))

    assert np.isclose(metrics["brier_binary"], expected_brier_binary)
    assert np.isclose(metrics["brier_recommended"], expected_brier_binary)
    assert np.isclose(metrics["brier"], expected_brier_binary)
    assert np.isfinite(metrics["ece_binary_pos"])
    assert np.isfinite(metrics["log_loss"])
    assert "binary_pos" in curves
    assert "top1" in curves


def test_probability_quality_multiclass_brier_normalization_and_ovr():
    y_true = ["a", "b", "c"]
    y_pred = ["a", "b", "a"]
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
        ],
        dtype=float,
    )

    metrics, curves = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["a", "b", "c"],
        bins=3,
        mode="multiclass",
    )

    assert np.isclose(metrics["brier_multiclass_mean"], metrics["brier_multiclass_sum"] / 3.0)
    assert np.isclose(metrics["brier_recommended"], metrics["brier_multiclass_mean"])
    # Top1 correctness must use argmax(proba) not y_pred.
    assert np.isclose(metrics["accuracy_top1"], 2.0 / 3.0)
    assert 0.0 <= metrics["ece_top1"] <= 1.0
    assert 0.0 <= metrics["ece_ovr_macro"] <= 1.0
    assert "top1" in curves
    assert "ovr_macro" in curves
    assert "ovr_a" in curves
    assert "ovr_b" in curves
    assert "ovr_c" in curves


def test_probability_quality_meta_uses_argmax_and_reports_mismatch():
    y_true = ["a", "b", "c", "c"]
    y_pred = ["a", "a", "c", "b"]  # intentionally differs from argmax on rows 2 and 4
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],  # argmax a
            [0.1, 0.8, 0.1],  # argmax b
            [0.1, 0.2, 0.7],  # argmax c
            [0.2, 0.1, 0.7],  # argmax c
        ],
        dtype=float,
    )

    metrics, curves = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["a", "b", "c"],
        bins=4,
        mode="meta",
    )

    assert metrics["pred_alignment_mismatch_rate"] > 0.0
    # Argmax predictions are all correct for this setup.
    assert np.isclose(metrics["accuracy_top1"], 1.0)
    assert "argmax(y_proba)" in metrics["pred_alignment_note"]
    assert "top1" in curves


def test_probability_quality_hierarchical_reports_mismatch_note():
    y_true = ["L1_A", "L1_B", "L1_C"]
    y_pred = ["L1_A", "L1_A", "L1_B"]
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],  # argmax L1_A
            [0.1, 0.8, 0.1],  # argmax L1_B
            [0.1, 0.2, 0.7],  # argmax L1_C
        ],
        dtype=float,
    )

    metrics, curves = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["L1_A", "L1_B", "L1_C"],
        bins=3,
        mode="hierarchical",
    )

    assert metrics["pred_alignment_mismatch_rate"] > 0.0
    assert (
        "hierarchical constraints alter the final predicted label" in metrics["pred_alignment_note"]
    )
    assert "top1" in curves
    assert "ovr_macro" in curves


def test_calibration_curve_quantile_with_duplicate_confidences():
    y_true = ["a"] * 12
    y_pred = ["a"] * 12
    # Repeated confidences trigger duplicated quantile edges.
    y_proba = np.array([[1.0, 0.0, 0.0]] * 8 + [[0.5, 0.3, 0.2]] * 4, dtype=float)

    metrics, curves = compute_probability_quality(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=["a", "b", "c"],
        bins=10,
        mode="multiclass",
        binning="quantile",
    )

    top1_curve = curves["top1"]
    assert not top1_curve.empty
    assert top1_curve["bin_start"].is_monotonic_increasing
    assert top1_curve["bin_end"].is_monotonic_increasing
    assert int(top1_curve["n"].sum()) == len(y_true)
    assert np.isfinite(metrics["ece_top1"])
