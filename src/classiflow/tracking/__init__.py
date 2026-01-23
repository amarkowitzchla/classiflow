"""
Experiment tracking integration for classiflow.

This module provides optional integration with experiment tracking systems
like MLflow and Weights & Biases. Tracking is disabled by default and
requires installing the appropriate optional dependencies.

Usage
-----
>>> from classiflow.tracking import get_tracker
>>> tracker = get_tracker("mlflow", experiment_name="my-experiment")
>>> with tracker.start_run(run_name="run-1"):
...     tracker.log_params({"lr": 0.01})
...     tracker.log_metrics({"accuracy": 0.95})

Installation
------------
For MLflow: pip install classiflow[mlflow]
For W&B: pip install classiflow[wandb]
For both: pip install classiflow[tracking]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from classiflow.tracking.base import ExperimentTracker
from classiflow.tracking.noop import NoOpTracker
from classiflow.tracking.utils import (
    extract_loggable_params,
    flatten_dict,
    sanitize_metric_name,
    summarize_metrics,
)

__all__ = [
    "ExperimentTracker",
    "NoOpTracker",
    "get_tracker",
    "extract_loggable_params",
    "flatten_dict",
    "sanitize_metric_name",
    "summarize_metrics",
]

logger = logging.getLogger(__name__)


def get_tracker(
    backend: Optional[str] = None,
    experiment_name: Optional[str] = None,
    **kwargs: Any,
) -> ExperimentTracker:
    """
    Get an experiment tracker instance.

    Factory function that returns the appropriate tracker based on
    the backend parameter. Returns a NoOpTracker if no backend is
    specified or if the backend is not available.

    Parameters
    ----------
    backend : str, optional
        Tracking backend to use. Options: "mlflow", "wandb", None.
        If None, returns a NoOpTracker.
    experiment_name : str, optional
        Name for the experiment/project. Defaults vary by backend.
    **kwargs
        Additional arguments passed to the tracker constructor.
        For MLflow: tracking_uri
        For W&B: project, entity, group

    Returns
    -------
    ExperimentTracker
        Configured tracker instance

    Raises
    ------
    ImportError
        If the requested backend is not installed
    ValueError
        If an unknown backend is specified

    Examples
    --------
    >>> # No tracking (default)
    >>> tracker = get_tracker()
    >>> isinstance(tracker, NoOpTracker)
    True

    >>> # MLflow tracking
    >>> tracker = get_tracker("mlflow", experiment_name="my-exp")

    >>> # W&B tracking
    >>> tracker = get_tracker("wandb", experiment_name="my-project")
    """
    if backend is None or backend.lower() == "none":
        return NoOpTracker()

    backend_lower = backend.lower()

    if backend_lower == "mlflow":
        try:
            from classiflow.tracking.mlflow_tracker import MLflowTracker
        except ImportError as e:
            raise ImportError(
                "MLflow is not installed. Install with: pip install classiflow[mlflow]"
            ) from e
        return MLflowTracker(
            experiment_name=experiment_name or "classiflow",
            **kwargs,
        )

    if backend_lower == "wandb":
        try:
            from classiflow.tracking.wandb_tracker import WandBTracker
        except ImportError as e:
            raise ImportError(
                "Weights & Biases is not installed. Install with: pip install classiflow[wandb]"
            ) from e
        return WandBTracker(
            project=experiment_name or "classiflow",
            **kwargs,
        )

    raise ValueError(
        f"Unknown tracking backend: {backend}. "
        f"Supported backends: mlflow, wandb"
    )
