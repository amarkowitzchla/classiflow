"""Tests for W&B tracker (mocked)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# Skip all tests if wandb is not installed
wandb = pytest.importorskip("wandb")

from classiflow.tracking.wandb_tracker import WandBTracker


class TestWandBTracker:
    """Tests for WandBTracker with mocked wandb calls."""

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_init_stores_project(self, mock_wandb):
        """Tracker init should store project name."""
        tracker = WandBTracker(project="test-project")
        assert tracker.project == "test-project"

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_init_stores_entity(self, mock_wandb):
        """Tracker init should store entity if provided."""
        tracker = WandBTracker(project="test-project", entity="my-team")
        assert tracker.entity == "my-team"

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_start_run_calls_init(self, mock_wandb):
        """start_run should call wandb.init."""
        tracker = WandBTracker(project="test-project")
        tracker.start_run(run_name="test-run")
        mock_wandb.init.assert_called_once()

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_start_run_with_tags(self, mock_wandb):
        """start_run should convert tags dict to list format."""
        tracker = WandBTracker(project="test-project")
        tracker.start_run(run_name="test-run", tags={"key": "value"})
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["tags"] == ["key:value"]

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_start_run_with_group(self, mock_wandb):
        """start_run should pass group if configured."""
        tracker = WandBTracker(project="test-project", group="my-group")
        tracker.start_run(run_name="test-run")
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["group"] == "my-group"

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_end_run(self, mock_wandb):
        """end_run should call run.finish."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.end_run()
        mock_run.finish.assert_called_once()

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_params(self, mock_wandb):
        """log_params should update run config."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_params({"lr": 0.01, "epochs": 100})
        mock_run.config.update.assert_called_once()

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_params_flattens_nested(self, mock_wandb):
        """log_params should flatten nested dicts with . separator."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_params({"outer": {"inner": 1}})
        call_args = mock_run.config.update.call_args[0][0]
        # W&B uses . separator by default
        assert "outer.inner" in call_args

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_metrics(self, mock_wandb):
        """log_metrics should call run.log."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
        mock_run.log.assert_called_once()

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_metrics_with_step(self, mock_wandb):
        """log_metrics should pass step parameter."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95}, step=10)
        call_kwargs = mock_run.log.call_args[1]
        assert call_kwargs["step"] == 10

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_metrics_filters_non_numeric(self, mock_wandb):
        """log_metrics should filter non-numeric values."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_metrics({"accuracy": 0.95, "model": "lr"})
        call_kwargs = mock_run.log.call_args[1]
        assert "accuracy" in call_kwargs["data"]
        assert "model" not in call_kwargs["data"]

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_artifact(self, mock_wandb, tmp_path):
        """log_artifact should create and log wandb Artifact."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_artifact(test_file)

        mock_wandb.Artifact.assert_called_once()
        mock_artifact.add_file.assert_called_once_with(str(test_file))
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_artifact_missing_file_warns(self, mock_wandb, tmp_path, caplog):
        """log_artifact should warn for missing files."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_artifact(tmp_path / "nonexistent.txt")
        mock_run.log_artifact.assert_not_called()
        assert "not found" in caplog.text

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_log_figure(self, mock_wandb):
        """log_figure should call run.log with wandb.Image."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_image = MagicMock()
        mock_wandb.Image.return_value = mock_image

        mock_figure = MagicMock()

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.log_figure("test_figure", mock_figure)

        mock_wandb.Image.assert_called_once_with(mock_figure)
        mock_run.log.assert_called_once_with({"test_figure": mock_image})

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_set_tags(self, mock_wandb):
        """set_tags should update run.tags."""
        mock_run = MagicMock()
        mock_run.tags = []
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        tracker.start_run()
        tracker.set_tags({"task_type": "binary"})
        assert "task_type:binary" in mock_run.tags

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_context_manager(self, mock_wandb):
        """Tracker should work as context manager."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        tracker = WandBTracker(project="test-project")
        with tracker.start_run(run_name="test"):
            tracker.log_params({"a": 1})
        mock_run.finish.assert_called_once()

    @patch("classiflow.tracking.wandb_tracker.wandb")
    def test_operations_without_run_warn(self, mock_wandb, caplog):
        """Operations without active run should warn."""
        tracker = WandBTracker(project="test-project")
        # Don't start a run
        tracker.log_params({"a": 1})
        assert "No active W&B run" in caplog.text
