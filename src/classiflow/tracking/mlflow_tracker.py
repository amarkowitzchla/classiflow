"""MLflow experiment tracking backend."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.figure

from classiflow.tracking.base import ExperimentTracker
from classiflow.tracking.utils import flatten_dict, sanitize_metric_name

logger = logging.getLogger(__name__)

# Import mlflow at module level to fail fast if not installed
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


class MLflowTracker(ExperimentTracker):
    """
    MLflow experiment tracking backend.

    Logs experiments to MLflow tracking server or local filesystem.
    Supports logging parameters, metrics, artifacts, and figures.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
    tracking_uri : str, optional
        MLflow tracking server URI. Defaults to local "./mlruns"

    Examples
    --------
    >>> tracker = MLflowTracker("my-experiment")
    >>> with tracker.start_run(run_name="run-1"):
    ...     tracker.log_params({"learning_rate": 0.01})
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_artifact("model.pkl")
    """

    def __init__(
        self,
        experiment_name: str = "classiflow",
        tracking_uri: Optional[str] = None,
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Install with: pip install classiflow[mlflow]"
            )

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_active = False

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "MLflowTracker":
        """Start a new MLflow run."""
        if self._run_active:
            logger.warning("MLflow run already active, ending previous run")
            self.end_run()

        mlflow.start_run(run_name=run_name)
        self._run_active = True

        if tags:
            self.set_tags(tags)

        logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
        return self

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._run_active:
            mlflow.end_run()
            self._run_active = False
            logger.debug("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self._run_active:
            logger.warning("No active MLflow run, skipping log_params")
            return

        # Flatten nested dicts and convert values to strings
        flat_params = flatten_dict(params)
        safe_params = {}
        for k, v in flat_params.items():
            # MLflow has a 500 char limit for param values
            str_val = str(v)
            if len(str_val) > 500:
                str_val = str_val[:497] + "..."
            # MLflow param names can't contain certain characters
            safe_key = sanitize_metric_name(k)
            safe_params[safe_key] = str_val

        try:
            mlflow.log_params(safe_params)
            logger.debug(f"Logged {len(safe_params)} params to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow."""
        if not self._run_active:
            logger.warning("No active MLflow run, skipping log_metrics")
            return

        # Flatten and sanitize metric names
        flat_metrics = flatten_dict(metrics)
        safe_metrics = {}
        for k, v in flat_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                safe_key = sanitize_metric_name(k)
                safe_metrics[safe_key] = float(v)

        try:
            mlflow.log_metrics(safe_metrics, step=step)
            logger.debug(f"Logged {len(safe_metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, path: Union[str, Path]) -> None:
        """Log a file artifact to MLflow."""
        if not self._run_active:
            logger.warning("No active MLflow run, skipping log_artifact")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact not found: {path}")
            return

        try:
            mlflow.log_artifact(str(path))
            logger.debug(f"Logged artifact: {path.name}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_figure(
        self,
        name: str,
        figure: matplotlib.figure.Figure,
    ) -> None:
        """Log a matplotlib figure to MLflow."""
        if not self._run_active:
            logger.warning("No active MLflow run, skipping log_figure")
            return

        try:
            # Save to temp file and log
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                figure.savefig(f.name, dpi=150, bbox_inches="tight")
                mlflow.log_artifact(f.name, artifact_path="figures")
                # Clean up
                Path(f.name).unlink(missing_ok=True)
            logger.debug(f"Logged figure: {name}")
        except Exception as e:
            logger.warning(f"Failed to log figure {name}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current MLflow run."""
        if not self._run_active:
            logger.warning("No active MLflow run, skipping set_tags")
            return

        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags on MLflow run")
        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def __enter__(self) -> "MLflowTracker":
        """Context manager entry - run should already be started."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ends the run."""
        self.end_run()
        return None
