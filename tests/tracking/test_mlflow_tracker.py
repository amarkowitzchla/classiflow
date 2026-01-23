"""Tests for MLflow tracker (mocked)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# Skip all tests if mlflow is not installed
mlflow = pytest.importorskip("mlflow")

from classiflow.tracking.mlflow_tracker import MLflowTracker


class TestMLflowTracker:
    """Tests for MLflowTracker with mocked mlflow calls."""

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_init_sets_experiment(self, mock_mlflow):
        """Tracker init should set the experiment name."""
        tracker = MLflowTracker(experiment_name="test-exp")
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_init_with_tracking_uri(self, mock_mlflow):
        """Tracker should set tracking URI if provided."""
        tracker = MLflowTracker(
            experiment_name="test-exp",
            tracking_uri="http://localhost:5000",
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_start_run(self, mock_mlflow):
        """start_run should call mlflow.start_run."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run(run_name="test-run")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_start_run_with_tags(self, mock_mlflow):
        """start_run should set tags if provided."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run(run_name="test-run", tags={"key": "value"})
        mock_mlflow.set_tags.assert_called_once_with({"key": "value"})

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_end_run(self, mock_mlflow):
        """end_run should call mlflow.end_run."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_params(self, mock_mlflow):
        """log_params should call mlflow.log_params."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_params({"lr": 0.01, "epochs": 100})
        mock_mlflow.log_params.assert_called_once()

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_params_flattens_nested(self, mock_mlflow):
        """log_params should flatten nested dicts."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_params({"outer": {"inner": 1}})
        # Check that flattened params were logged
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert "outer/inner" in call_args

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_params_truncates_long_values(self, mock_mlflow):
        """log_params should truncate values > 500 chars."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        long_value = "x" * 600
        tracker.log_params({"long": long_value})
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert len(call_args["long"]) == 500

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_metrics(self, mock_mlflow):
        """log_metrics should call mlflow.log_metrics."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
        mock_mlflow.log_metrics.assert_called_once()

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_metrics_with_step(self, mock_mlflow):
        """log_metrics should pass step parameter."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95}, step=10)
        mock_mlflow.log_metrics.assert_called_once_with(ANY, step=10)

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_metrics_filters_non_numeric(self, mock_mlflow):
        """log_metrics should filter non-numeric values."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95, "model": "lr", "converged": True})
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert "accuracy" in call_args
        assert "model" not in call_args
        assert "converged" not in call_args

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_artifact(self, mock_mlflow, tmp_path):
        """log_artifact should call mlflow.log_artifact."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_artifact(test_file)
        mock_mlflow.log_artifact.assert_called_once_with(str(test_file))

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_log_artifact_missing_file_warns(self, mock_mlflow, tmp_path, caplog):
        """log_artifact should warn for missing files."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.log_artifact(tmp_path / "nonexistent.txt")
        mock_mlflow.log_artifact.assert_not_called()
        assert "not found" in caplog.text

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_set_tags(self, mock_mlflow):
        """set_tags should call mlflow.set_tags."""
        tracker = MLflowTracker(experiment_name="test-exp")
        tracker.start_run()
        tracker.set_tags({"task_type": "binary"})
        # Called once in start_run (if tags provided) or once here
        mock_mlflow.set_tags.assert_called()

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_context_manager(self, mock_mlflow):
        """Tracker should work as context manager."""
        tracker = MLflowTracker(experiment_name="test-exp")
        with tracker.start_run(run_name="test"):
            tracker.log_params({"a": 1})
        mock_mlflow.end_run.assert_called_once()

    @patch("classiflow.tracking.mlflow_tracker.mlflow")
    def test_operations_without_run_warn(self, mock_mlflow, caplog):
        """Operations without active run should warn."""
        tracker = MLflowTracker(experiment_name="test-exp")
        # Don't start a run
        tracker.log_params({"a": 1})
        assert "No active MLflow run" in caplog.text
        mock_mlflow.log_params.assert_not_called()
