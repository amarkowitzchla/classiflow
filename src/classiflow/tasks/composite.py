"""Load composite tasks from JSON specification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from classiflow.tasks.builder import TaskBuilder

logger = logging.getLogger(__name__)


def load_composite_tasks(
    json_path: Path,
    builder: TaskBuilder,
) -> TaskBuilder:
    """
    Load composite tasks from JSON and add to builder.

    Supported JSON formats:

    1) Dict form (preferred):
       {
         "WNT_like": {"pos": ["WNT"], "neg": "rest"},
         "G3_vs_G4": {"pos": ["G3"], "neg": ["G4"]}
       }

    2) List form:
       [
         {"name": "WNT_like", "pos": ["WNT"], "neg": "rest"},
         {"name": "G3_vs_G4", "pos": ["G3"], "neg": ["G4"]}
       ]

    Parameters
    ----------
    json_path : Path
        Path to tasks JSON file
    builder : TaskBuilder
        TaskBuilder instance to add tasks to

    Returns
    -------
    builder : TaskBuilder
        Builder with added composite tasks
    """
    logger.info(f"Loading composite tasks from {json_path}")

    with open(json_path, "r") as f:
        spec = json.load(f)

    for name, rule in _iter_task_rules(spec):
        if isinstance(rule, list):
            # Tolerate list -> treat as pos classes, neg="rest"
            pos = rule
            neg = "rest"
        elif isinstance(rule, dict):
            pos = rule.get("pos", [])
            neg = rule.get("neg", "rest")
        else:
            logger.warning(f"Skipping invalid task rule for '{name}': {rule}")
            continue

        if not pos:
            logger.warning(f"Skipping task '{name}': no positive classes specified")
            continue

        builder.add_composite_task(name, pos_classes=pos, neg_classes=neg)

    logger.info(f"Loaded {len(builder.tasks)} total tasks")
    return builder


def _iter_task_rules(spec: Any):
    """
    Yield (name, rule) pairs from various JSON formats.

    Yields
    ------
    name : str
        Task name
    rule : dict or list
        Task rule specification
    """
    if isinstance(spec, dict):
        # Dict form: {name: rule, ...}
        for nm, rl in spec.items():
            yield nm, rl
    elif isinstance(spec, list):
        # List form: [{name: ..., pos: ..., neg: ...}, ...]
        for item in spec:
            if isinstance(item, dict):
                nm = item.get("name") or item.get("task")
                rl = item
                if not nm:
                    # Infer name from pos/neg
                    nm = _infer_task_name(item)
                yield nm, rl
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # 2-tuple form: [name, rule]
                nm, rl = item[0], item[1]
                yield nm, rl
    else:
        logger.warning(f"Unknown task spec format: {type(spec)}")


def _infer_task_name(rule: Dict[str, Any]) -> str:
    """Infer task name from pos/neg specification."""
    pos = rule.get("pos", [])
    neg = rule.get("neg", "rest")
    pos_str = "_".join(map(str, pos))
    if isinstance(neg, (list, tuple, set)):
        neg_str = "_".join(map(str, neg))
        return f"{pos_str}_vs_{neg_str}"
    elif neg == "rest":
        return f"{pos_str}_vs_Rest"
    else:
        return f"{pos_str}_vs_{neg}"
