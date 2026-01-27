"""Tests for plot data export functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from classiflow.plots.data_export import (
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_averaged_roc_data,
    compute_averaged_pr_data,
    save_plot_data,
    create_plot_manifest,
    generate_inference_plots,
)
from classiflow.plots.schemas import PlotScope, PlotType, TaskType, PlotKey


class TestComputeRocCurveData:
    """Tests for compute_roc_curve_data function."""

    def test_binary_classification(self):
        """Test ROC computation for binary classification."""
        # Simple binary classification data
        y_true = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
        ])
        classes = ["Negative", "Positive"]

        result = compute_roc_curve_data(
            y_true, y_proba, classes, "test_run_001",
            scope=PlotScope.INFERENCE,
        )

        assert result.plot_type == PlotType.ROC
        assert result.scope == PlotScope.INFERENCE
        assert result.task == TaskType.BINARY
        assert result.labels == classes
        assert len(result.curves) >= 1
        assert result.metadata.run_id == "test_run_001"

        # Check that AUC is computed
        assert result.summary.auc is not None
        assert len(result.summary.auc) > 0

        # Check curve data structure
        for curve in result.curves:
            assert len(curve.x) > 0
            assert len(curve.y) > 0
            assert len(curve.x) == len(curve.y)
            # ROC: x should be in [0, 1]
            assert all(0 <= x <= 1 for x in curve.x)
            assert all(0 <= y <= 1 for y in curve.y)

    def test_multiclass_classification(self):
        """Test ROC computation for multiclass classification."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        classes = ["Class_A", "Class_B", "Class_C"]

        y_true = np.random.randint(0, n_classes, n_samples)
        y_proba = np.random.dirichlet([1] * n_classes, n_samples)

        result = compute_roc_curve_data(
            y_true, y_proba, classes, "test_run_002",
            scope=PlotScope.FOLD, fold=1,
        )

        assert result.plot_type == PlotType.ROC
        assert result.task == TaskType.MULTICLASS
        assert result.labels == classes
        assert result.metadata.fold == 1

        # Should have per-class curves plus micro average
        curve_labels = [c.label for c in result.curves]
        assert "micro" in curve_labels

        # Check AUC values
        assert result.summary.auc is not None
        assert "micro" in result.summary.auc

    def test_include_thresholds(self):
        """Test that thresholds are included when requested."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])
        classes = ["Neg", "Pos"]

        result_with = compute_roc_curve_data(
            y_true, y_proba, classes, "test",
            include_thresholds=True,
        )
        result_without = compute_roc_curve_data(
            y_true, y_proba, classes, "test",
            include_thresholds=False,
        )

        assert result_with.curves[0].thresholds is not None
        assert result_without.curves[0].thresholds is None


class TestComputePrCurveData:
    """Tests for compute_pr_curve_data function."""

    def test_binary_classification(self):
        """Test PR computation for binary classification."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
        ])
        classes = ["Negative", "Positive"]

        result = compute_pr_curve_data(
            y_true, y_proba, classes, "test_run",
        )

        assert result.plot_type == PlotType.PR
        assert result.task == TaskType.BINARY

        # Check AP is computed
        assert result.summary.ap is not None
        assert len(result.summary.ap) > 0

        # Check curve values
        for curve in result.curves:
            assert len(curve.x) > 0
            assert len(curve.y) > 0
            # PR: x (recall) and y (precision) should be in [0, 1]
            assert all(0 <= x <= 1 for x in curve.x)
            assert all(0 <= y <= 1 for y in curve.y)

    def test_multiclass_classification(self):
        """Test PR computation for multiclass classification."""
        np.random.seed(42)
        n_samples = 100
        classes = ["A", "B", "C"]

        y_true = np.random.randint(0, 3, n_samples)
        y_proba = np.random.dirichlet([1, 1, 1], n_samples)

        result = compute_pr_curve_data(
            y_true, y_proba, classes, "test",
        )

        assert result.task == TaskType.MULTICLASS
        curve_labels = [c.label for c in result.curves]
        assert "micro" in curve_labels


class TestAveragedCurves:
    """Tests for averaged curve computation."""

    def test_averaged_roc(self):
        """Test averaged ROC curve computation across folds."""
        # Simulate 3 folds with simple curves
        all_fpr = [
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.4, 1.0]),
            np.array([0.0, 0.6, 1.0]),
        ]
        all_tpr = [
            np.array([0.0, 0.8, 1.0]),
            np.array([0.0, 0.85, 1.0]),
            np.array([0.0, 0.75, 1.0]),
        ]
        all_aucs = [0.9, 0.92, 0.88]
        classes = ["Neg", "Pos"]

        result = compute_averaged_roc_data(
            all_fpr, all_tpr, all_aucs, classes, "test",
        )

        assert result.scope == PlotScope.AVERAGED
        assert len(result.curves) >= 1
        assert result.curves[0].label == "mean"

        # Check std band
        assert result.std_band is not None
        assert "x" in result.std_band
        assert "y_upper" in result.std_band
        assert "y_lower" in result.std_band

        # Check fold curves
        assert result.fold_curves is not None
        assert len(result.fold_curves) == 3

        # Check fold metrics
        assert result.fold_metrics is not None
        assert "auc" in result.fold_metrics
        assert len(result.fold_metrics["auc"]) == 3

        # Check summary
        assert result.summary.auc is not None
        assert "mean" in result.summary.auc
        assert "std" in result.summary.auc

    def test_averaged_pr(self):
        """Test averaged PR curve computation across folds."""
        all_rec = [
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.6, 1.0]),
            np.array([0.0, 0.4, 1.0]),
        ]
        all_prec = [
            np.array([1.0, 0.8, 0.6]),
            np.array([1.0, 0.85, 0.65]),
            np.array([1.0, 0.75, 0.55]),
        ]
        all_aps = [0.85, 0.88, 0.82]
        classes = ["A", "B"]

        result = compute_averaged_pr_data(
            all_rec, all_prec, all_aps, classes, "test",
        )

        assert result.plot_type == PlotType.PR
        assert result.scope == PlotScope.AVERAGED

        # Check summary
        assert result.summary.ap is not None
        assert "mean" in result.summary.ap


class TestSavePlotData:
    """Tests for save_plot_data function."""

    def test_save_and_load(self):
        """Test saving plot data to JSON and loading it back."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])
        classes = ["Neg", "Pos"]

        plot_data = compute_roc_curve_data(
            y_true, y_proba, classes, "test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plots" / "roc_test.json"
            save_plot_data(plot_data, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["plot_type"] == "roc"
            assert loaded["scope"] == "inference"
            assert "curves" in loaded
            assert len(loaded["curves"]) > 0


class TestCreatePlotManifest:
    """Tests for create_plot_manifest function."""

    def test_create_manifest(self):
        """Test creating a plot manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            available = {
                PlotKey.ROC_AVERAGED: "plots/roc_averaged.json",
                PlotKey.PR_AVERAGED: "plots/pr_averaged.json",
            }
            fallback = {
                PlotKey.ROC_AVERAGED: "averaged_roc.png",
            }

            manifest = create_plot_manifest(
                run_dir, "test_run", available, fallback,
            )

            assert manifest.available == available
            assert manifest.fallback_pngs == fallback

            # Check file was created
            manifest_path = run_dir / "plots" / "plot_manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                loaded = json.load(f)

            assert loaded["available"] == available
            assert loaded["fallback_pngs"] == fallback


class TestGenerateInferencePlots:
    """Tests for generate_inference_plots function."""

    def test_generate_inference_plots(self):
        """Test generating all inference plots."""
        np.random.seed(42)
        n_samples = 50
        classes = ["Normal", "Tumor"]

        y_true = np.random.randint(0, 2, n_samples)
        y_proba = np.random.dirichlet([1, 1], n_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            manifest = generate_inference_plots(
                run_dir, "test_run", y_true, y_proba, classes,
            )

            # Check manifest
            assert PlotKey.ROC_INFERENCE in manifest.available
            assert PlotKey.PR_INFERENCE in manifest.available

            # Check files exist
            roc_path = run_dir / "plots" / "roc_inference.json"
            pr_path = run_dir / "plots" / "pr_inference.json"
            manifest_path = run_dir / "plots" / "plot_manifest.json"

            assert roc_path.exists()
            assert pr_path.exists()
            assert manifest_path.exists()

            # Verify ROC data
            with open(roc_path) as f:
                roc_data = json.load(f)
            assert roc_data["plot_type"] == "roc"
            assert roc_data["scope"] == "inference"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_class_in_fold(self):
        """Test handling when a fold has only one class."""
        # This can happen with small datasets
        y_true = np.array([0, 0, 0, 0])  # Only one class
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ])
        classes = ["Neg", "Pos"]

        # Should not raise an error
        result = compute_roc_curve_data(
            y_true, y_proba, classes, "test",
        )
        assert result is not None

    def test_empty_curves_handled(self):
        """Test that empty curve data raises appropriate error."""
        all_fpr = []
        all_tpr = []
        all_aucs = []
        classes = ["A", "B"]

        # Empty input should raise an error (either ValueError, IndexError, or TypeError)
        with pytest.raises((ValueError, IndexError, TypeError)):
            compute_averaged_roc_data(
                all_fpr, all_tpr, all_aucs, classes, "test",
            )

    def test_valid_data_with_no_nans(self):
        """Test that valid data without NaN works correctly."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([
            [0.8, 0.2],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.2, 0.8],
        ])
        classes = ["Neg", "Pos"]

        result = compute_roc_curve_data(
            y_true, y_proba, classes, "test",
        )
        # Curves should not contain NaN
        for curve in result.curves:
            assert all(np.isfinite(x) for x in curve.x)
            assert all(np.isfinite(y) for y in curve.y)
