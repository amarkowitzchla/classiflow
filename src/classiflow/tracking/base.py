"""Base class for experiment tracking backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.figure


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking backends.

    Provides a unified interface for logging experiments to various
    tracking systems (MLflow, W&B, etc.). All methods are designed
    to be non-blocking and fail gracefully.

    Usage
    -----
    >>> tracker = get_tracker("mlflow", experiment_name="my-experiment")
    >>> with tracker.start_run(run_name="run-1"):
    ...     tracker.log_params({"lr": 0.01, "epochs": 100})
    ...     tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    ...     tracker.log_artifact("model.pkl")
    """

    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "ExperimentTracker":
        """
        Start a new tracking run.

        Parameters
        ----------
        run_name : str, optional
            Name for this run
        tags : dict, optional
            Tags to associate with the run

        Returns
        -------
        self : ExperimentTracker
            Returns self for context manager usage
        """
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current tracking run."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters/hyperparameters for the run.

        Parameters
        ----------
        params : dict
            Parameter name-value pairs. Values will be converted to strings.
            Nested dicts will be flattened with "/" separator.
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics for the run.

        Parameters
        ----------
        metrics : dict
            Metric name-value pairs. Nested dicts will be flattened.
        step : int, optional
            Step/epoch number for the metrics
        """
        pass

    @abstractmethod
    def log_artifact(self, path: Union[str, Path]) -> None:
        """
        Log a file artifact.

        Parameters
        ----------
        path : str or Path
            Path to the artifact file
        """
        pass

    @abstractmethod
    def log_figure(
        self,
        name: str,
        figure: matplotlib.figure.Figure,
    ) -> None:
        """
        Log a matplotlib figure.

        Parameters
        ----------
        name : str
            Name for the figure
        figure : matplotlib.figure.Figure
            The figure to log
        """
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags on the current run.

        Parameters
        ----------
        tags : dict
            Tag name-value pairs
        """
        pass

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ends the run."""
        self.end_run()
        return None
