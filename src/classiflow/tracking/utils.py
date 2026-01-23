"""Utility functions for experiment tracking."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to flatten
    parent_key : str
        Prefix for keys (used in recursion)
    sep : str
        Separator between nested keys

    Returns
    -------
    dict
        Flattened dictionary with joined keys

    Examples
    --------
    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {"a/b": 1, "a/c": 2, "d": 3}
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sanitize_metric_name(name: str) -> str:
    """
    Sanitize a metric name for tracking systems.

    Replaces invalid characters with underscores and ensures
    the name is valid for both MLflow and W&B.

    Parameters
    ----------
    name : str
        Original metric name

    Returns
    -------
    str
        Sanitized metric name

    Examples
    --------
    >>> sanitize_metric_name("fold[1]/accuracy")
    "fold_1_/accuracy"
    """
    # Replace brackets with underscores (common in fold names)
    name = re.sub(r"[\[\]]", "_", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove any characters that aren't alphanumeric, underscore, dash, dot, or slash
    name = re.sub(r"[^a-zA-Z0-9_\-./]", "", name)
    return name


def extract_loggable_params(config: Any) -> Dict[str, Any]:
    """
    Extract loggable parameters from a config object.

    Handles dataclasses and dicts, converting Path objects to strings
    and flattening nested structures.

    Parameters
    ----------
    config : dataclass or dict
        Configuration object

    Returns
    -------
    dict
        Flattened dictionary of parameters suitable for logging

    Examples
    --------
    >>> from classiflow.config import TrainConfig
    >>> config = TrainConfig(label_col="target", outer_folds=5)
    >>> params = extract_loggable_params(config)
    >>> params["label_col"]
    "target"
    >>> params["outer_folds"]
    5
    """
    if is_dataclass(config) and not isinstance(config, type):
        d = asdict(config)
    elif isinstance(config, dict):
        d = config.copy()
    elif hasattr(config, "to_dict"):
        d = config.to_dict()
    else:
        raise TypeError(f"Cannot extract params from {type(config)}")

    return _convert_values(d)


def _convert_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert values in a dict to loggable types.

    - Path -> str
    - None -> "None"
    - list -> comma-separated string
    - nested dict -> flattened
    """
    result = {}
    for k, v in d.items():
        if v is None:
            result[k] = "None"
        elif isinstance(v, Path):
            result[k] = str(v)
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                result[k] = "[]"
            elif all(isinstance(x, (str, int, float, bool)) for x in v):
                result[k] = ",".join(str(x) for x in v)
            else:
                result[k] = str(v)
        elif isinstance(v, dict):
            # Flatten nested dicts
            for nested_k, nested_v in _convert_values(v).items():
                result[f"{k}/{nested_k}"] = nested_v
        else:
            result[k] = v
    return result


def summarize_metrics(
    metrics_dict: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Extract numeric metrics from a results dictionary.

    Filters to only float/int values and optionally adds a prefix.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary potentially containing metrics
    prefix : str
        Prefix to add to metric names

    Returns
    -------
    dict
        Dictionary of numeric metrics only
    """
    result = {}
    flat = flatten_dict(metrics_dict) if any(isinstance(v, dict) for v in metrics_dict.values()) else metrics_dict

    for k, v in flat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            key = f"{prefix}{k}" if prefix else k
            result[sanitize_metric_name(key)] = float(v)

    return result
