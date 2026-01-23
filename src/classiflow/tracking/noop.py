"""No-op tracker implementation (null object pattern)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.figure

from classiflow.tracking.base import ExperimentTracker


class NoOpTracker(ExperimentTracker):
    """
    No-operation tracker that silently ignores all tracking calls.

    This is the default tracker when no tracking backend is configured.
    All methods are no-ops that return immediately, ensuring training
    code works identically with or without tracking enabled.

    Usage
    -----
    >>> tracker = NoOpTracker()
    >>> with tracker.start_run():
    ...     tracker.log_params({"lr": 0.01})  # Does nothing
    ...     tracker.log_metrics({"accuracy": 0.95})  # Does nothing
    """

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "NoOpTracker":
        """No-op: returns self for context manager."""
        return self

    def end_run(self) -> None:
        """No-op: does nothing."""
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """No-op: does nothing."""
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """No-op: does nothing."""
        pass

    def log_artifact(self, path: Union[str, Path]) -> None:
        """No-op: does nothing."""
        pass

    def log_figure(
        self,
        name: str,
        figure: matplotlib.figure.Figure,
    ) -> None:
        """No-op: does nothing."""
        pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        """No-op: does nothing."""
        pass
