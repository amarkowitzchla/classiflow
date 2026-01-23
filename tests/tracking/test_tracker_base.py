"""Tests for base tracker interface and no-op tracker."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from classiflow.tracking.base import ExperimentTracker
from classiflow.tracking.noop import NoOpTracker
from classiflow.tracking import get_tracker


class TestNoOpTracker:
    """Tests for NoOpTracker (null object pattern)."""

    def test_noop_tracker_is_experiment_tracker(self):
        """NoOpTracker should implement ExperimentTracker interface."""
        tracker = NoOpTracker()
        assert isinstance(tracker, ExperimentTracker)

    def test_start_run_returns_self(self):
        """start_run should return self for context manager usage."""
        tracker = NoOpTracker()
        result = tracker.start_run(run_name="test")
        assert result is tracker

    def test_context_manager_protocol(self):
        """NoOpTracker should work as a context manager."""
        tracker = NoOpTracker()
        with tracker.start_run(run_name="test") as t:
            assert t is tracker

    def test_log_params_does_nothing(self):
        """log_params should not raise errors."""
        tracker = NoOpTracker()
        with tracker.start_run():
            tracker.log_params({"lr": 0.01, "epochs": 100})

    def test_log_metrics_does_nothing(self):
        """log_metrics should not raise errors."""
        tracker = NoOpTracker()
        with tracker.start_run():
            tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
            tracker.log_metrics({"accuracy": 0.96}, step=1)

    def test_log_artifact_does_nothing(self, tmp_path):
        """log_artifact should not raise errors."""
        tracker = NoOpTracker()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with tracker.start_run():
            tracker.log_artifact(test_file)
            tracker.log_artifact(str(test_file))

    def test_log_figure_does_nothing(self):
        """log_figure should not raise errors."""
        tracker = NoOpTracker()
        mock_figure = MagicMock()

        with tracker.start_run():
            tracker.log_figure("test_figure", mock_figure)

    def test_set_tags_does_nothing(self):
        """set_tags should not raise errors."""
        tracker = NoOpTracker()
        with tracker.start_run():
            tracker.set_tags({"task_type": "binary", "backend": "sklearn"})

    def test_end_run_does_nothing(self):
        """end_run should not raise errors."""
        tracker = NoOpTracker()
        tracker.start_run()
        tracker.end_run()


class TestGetTracker:
    """Tests for get_tracker factory function."""

    def test_none_backend_returns_noop(self):
        """None backend should return NoOpTracker."""
        tracker = get_tracker(backend=None)
        assert isinstance(tracker, NoOpTracker)

    def test_none_string_backend_returns_noop(self):
        """'none' string backend should return NoOpTracker."""
        tracker = get_tracker(backend="none")
        assert isinstance(tracker, NoOpTracker)

    def test_none_string_case_insensitive(self):
        """'None' and 'NONE' should also work."""
        tracker1 = get_tracker(backend="None")
        tracker2 = get_tracker(backend="NONE")
        assert isinstance(tracker1, NoOpTracker)
        assert isinstance(tracker2, NoOpTracker)

    def test_unknown_backend_raises_error(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            get_tracker(backend="unknown")

    def test_mlflow_import_error(self):
        """MLflow backend should raise ImportError if not installed."""
        # This test may pass or fail depending on whether mlflow is installed
        try:
            tracker = get_tracker(backend="mlflow")
            # If mlflow is installed, this is fine
            assert tracker is not None
        except ImportError as e:
            assert "mlflow" in str(e).lower()

    def test_wandb_import_error(self):
        """W&B backend should raise ImportError if not installed."""
        # This test may pass or fail depending on whether wandb is installed
        try:
            tracker = get_tracker(backend="wandb")
            # If wandb is installed, this is fine
            assert tracker is not None
        except ImportError as e:
            assert "wandb" in str(e).lower()

    def test_experiment_name_passed_to_tracker(self):
        """experiment_name should be passed to tracker."""
        tracker = get_tracker(backend=None, experiment_name="my-experiment")
        # NoOpTracker doesn't use experiment_name but shouldn't fail
        assert isinstance(tracker, NoOpTracker)
