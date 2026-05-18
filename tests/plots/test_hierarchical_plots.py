"""Regression tests for hierarchical plotting curve fallbacks."""

from __future__ import annotations

import numpy as np

from classiflow.plots import hierarchical as hplots


def _raise_index_error(*_args, **_kwargs):
    raise IndexError("index 33700 is out of bounds for axis 0 with size 33700")


def test_safe_roc_curve_falls_back_when_sklearn_raises(monkeypatch):
    monkeypatch.setattr(hplots, "roc_curve", _raise_index_error)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    fpr, tpr, thresholds = hplots.safe_roc_curve(y_true, y_score, context="test ROC")

    assert len(fpr) >= 2
    assert len(tpr) == len(fpr)
    assert len(thresholds) == len(fpr)
    assert np.isfinite(fpr).all()
    assert np.isfinite(tpr).all()
    assert np.isfinite(thresholds[1:]).all()


def test_safe_pr_curve_and_ap_fall_back_when_sklearn_raises(monkeypatch):
    monkeypatch.setattr(hplots, "precision_recall_curve", _raise_index_error)
    monkeypatch.setattr(hplots, "average_precision_score", _raise_index_error)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    precision, recall, thresholds = hplots.safe_precision_recall_curve(
        y_true, y_score, context="test PR"
    )
    ap = hplots.safe_average_precision_score(y_true, y_score, context="test AP")

    assert len(precision) == len(recall)
    assert len(thresholds) == len(precision) - 1
    assert np.isfinite(precision).all()
    assert np.isfinite(recall).all()
    assert np.isfinite(thresholds).all()
    assert 0.0 <= ap <= 1.0


def test_binary_plot_functions_do_not_crash_on_single_class(tmp_path):
    y_true = np.ones(6, dtype=int)
    y_proba = np.array(
        [
            [0.05, 0.95],
            [0.02, 0.98],
            [0.10, 0.90],
            [0.08, 0.92],
            [0.03, 0.97],
            [0.07, 0.93],
        ],
        dtype=float,
    )
    classes = ["PB_RB", "RB"]

    roc_path = tmp_path / "roc_single_class.png"
    pr_path = tmp_path / "pr_single_class.png"

    hplots.plot_roc_curve(y_true, y_proba, classes, "Single-Class ROC", roc_path)
    hplots.plot_pr_curve(y_true, y_proba, classes, "Single-Class PR", pr_path)

    assert roc_path.exists()
    assert pr_path.exists()
