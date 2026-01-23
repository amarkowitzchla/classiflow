"""Weights & Biases experiment tracking backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.figure

from classiflow.tracking.base import ExperimentTracker
from classiflow.tracking.utils import flatten_dict, sanitize_metric_name

logger = logging.getLogger(__name__)

# Import wandb at module level to fail fast if not installed
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


class WandBTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracking backend.

    Logs experiments to W&B cloud or local server.
    Supports logging parameters, metrics, artifacts, and figures.

    Parameters
    ----------
    project : str
        W&B project name
    entity : str, optional
        W&B entity (team or username). Uses default if not specified.
    group : str, optional
        Group name for organizing related runs
    mode : str, optional
        W&B mode: "online", "offline", or "disabled"

    Examples
    --------
    >>> tracker = WandBTracker(project="my-project")
    >>> with tracker.start_run(run_name="run-1"):
    ...     tracker.log_params({"learning_rate": 0.01})
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_artifact("model.pkl")
    """

    def __init__(
        self,
        project: str = "classiflow",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Weights & Biases is not installed. "
                "Install with: pip install classiflow[wandb]"
            )

        self.project = project
        self.entity = entity
        self.group = group
        self.mode = mode
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._run_active = False

        logger.info(f"W&B tracker initialized for project: {project}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "WandBTracker":
        """Start a new W&B run."""
        if self._run_active:
            logger.warning("W&B run already active, ending previous run")
            self.end_run()

        # Convert tags dict to list format for W&B
        tag_list = None
        if tags:
            tag_list = [f"{k}:{v}" for k, v in tags.items()]

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": run_name,
            "tags": tag_list,
            "reinit": True,
        }

        if self.entity:
            init_kwargs["entity"] = self.entity
        if self.group:
            init_kwargs["group"] = self.group
        if self.mode:
            init_kwargs["mode"] = self.mode

        self._run = wandb.init(**init_kwargs)
        self._run_active = True

        logger.info(f"Started W&B run: {run_name or 'unnamed'}")
        return self

    def end_run(self) -> None:
        """End the current W&B run."""
        if self._run_active and self._run is not None:
            self._run.finish()
            self._run = None
            self._run_active = False
            logger.debug("Ended W&B run")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters/config to W&B."""
        if not self._run_active or self._run is None:
            logger.warning("No active W&B run, skipping log_params")
            return

        # Flatten nested dicts
        flat_params = flatten_dict(params, sep=".")

        try:
            self._run.config.update(flat_params)
            logger.debug(f"Logged {len(flat_params)} params to W&B")
        except Exception as e:
            logger.warning(f"Failed to log params to W&B: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B."""
        if not self._run_active or self._run is None:
            logger.warning("No active W&B run, skipping log_metrics")
            return

        # Flatten and filter to numeric values
        flat_metrics = flatten_dict(metrics, sep="/")
        safe_metrics = {}
        for k, v in flat_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                safe_key = sanitize_metric_name(k)
                safe_metrics[safe_key] = float(v)

        try:
            log_kwargs: Dict[str, Any] = {"data": safe_metrics}
            if step is not None:
                log_kwargs["step"] = step
            self._run.log(**log_kwargs)
            logger.debug(f"Logged {len(safe_metrics)} metrics to W&B")
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")

    def log_artifact(self, path: Union[str, Path]) -> None:
        """Log a file artifact to W&B."""
        if not self._run_active or self._run is None:
            logger.warning("No active W&B run, skipping log_artifact")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact not found: {path}")
            return

        try:
            artifact = wandb.Artifact(
                name=path.stem,
                type="file",
            )
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)
            logger.debug(f"Logged artifact: {path.name}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_figure(
        self,
        name: str,
        figure: matplotlib.figure.Figure,
    ) -> None:
        """Log a matplotlib figure to W&B."""
        if not self._run_active or self._run is None:
            logger.warning("No active W&B run, skipping log_figure")
            return

        try:
            self._run.log({name: wandb.Image(figure)})
            logger.debug(f"Logged figure: {name}")
        except Exception as e:
            logger.warning(f"Failed to log figure {name}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current W&B run."""
        if not self._run_active or self._run is None:
            logger.warning("No active W&B run, skipping set_tags")
            return

        try:
            # W&B tags are a list, so we convert dict to list format
            tag_list = [f"{k}:{v}" for k, v in tags.items()]
            # Merge with existing tags
            existing_tags = list(self._run.tags) if self._run.tags else []
            self._run.tags = existing_tags + tag_list
            logger.debug(f"Set {len(tags)} tags on W&B run")
        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def __enter__(self) -> "WandBTracker":
        """Context manager entry - run should already be started."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ends the run."""
        self.end_run()
        return None
