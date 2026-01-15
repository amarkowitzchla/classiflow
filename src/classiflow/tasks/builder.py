"""Task builder for OvR, pairwise, and composite binary tasks."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, Callable, List, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TaskBuilder:
    """
    Build binary classification tasks from multiclass labels.

    Supports:
    - One-vs-Rest (OvR) tasks
    - Pairwise tasks
    - Composite tasks from JSON specification
    """

    def __init__(self, classes: List[str]):
        """
        Initialize task builder.

        Parameters
        ----------
        classes : List[str]
            List of class labels in desired order
        """
        self.classes = classes
        self.tasks: Dict[str, Callable[[pd.Series], pd.Series]] = {}

    def build_ovr_tasks(self) -> TaskBuilder:
        """
        Build One-vs-Rest tasks for each class.

        Returns
        -------
        self : TaskBuilder
            For method chaining
        """
        for c in self.classes:
            task_name = f"{c}_vs_Rest"
            # Use closure to capture c
            self.tasks[task_name] = self._make_ovr_fn(c)
            logger.debug(f"Added OvR task: {task_name}")
        return self

    def build_pairwise_tasks(self) -> TaskBuilder:
        """
        Build pairwise tasks for all class pairs.

        Returns
        -------
        self : TaskBuilder
            For method chaining
        """
        for a, b in combinations(self.classes, 2):
            task_name = f"{a}_vs_{b}"
            self.tasks[task_name] = self._make_pairwise_fn(a, b)
            logger.debug(f"Added pairwise task: {task_name}")
        return self

    def add_composite_task(
        self,
        name: str,
        pos_classes: List[str],
        neg_classes: List[str] | Literal["rest"],
    ) -> TaskBuilder:
        """
        Add a composite task.

        Parameters
        ----------
        name : str
            Task name
        pos_classes : List[str]
            Classes to treat as positive (label=1)
        neg_classes : List[str] or "rest"
            Classes to treat as negative (label=0); "rest" means all classes not in pos_classes

        Returns
        -------
        self : TaskBuilder
            For method chaining
        """
        cls_set = set(self.classes)
        pos_set = set(pos_classes) & cls_set

        if neg_classes == "rest":
            neg_set = cls_set - pos_set
        else:
            neg_set = set(neg_classes) & cls_set

        if not pos_set:
            logger.warning(f"Composite task '{name}': no valid positive classes")
            return self
        if not neg_set:
            logger.warning(f"Composite task '{name}': no valid negative classes")
            return self

        self.tasks[name] = self._make_composite_fn(pos_set, neg_set)
        logger.debug(f"Added composite task: {name} (pos={pos_set}, neg={neg_set})")
        return self

    def build_all_auto_tasks(self) -> TaskBuilder:
        """
        Build all automatic tasks (OvR + pairwise).

        Returns
        -------
        self : TaskBuilder
            For method chaining
        """
        self.build_ovr_tasks()
        self.build_pairwise_tasks()
        return self

    def get_tasks(self) -> Dict[str, Callable[[pd.Series], pd.Series]]:
        """
        Get the built task dictionary.

        Returns
        -------
        tasks : Dict[str, Callable]
            Task name -> labeling function
        """
        return self.tasks

    @staticmethod
    def _make_ovr_fn(target_class: str) -> Callable[[pd.Series], pd.Series]:
        """Create OvR labeling function."""

        def labeler(y: pd.Series) -> pd.Series:
            return (y == target_class).astype(float)

        return labeler

    @staticmethod
    def _make_pairwise_fn(pos_class: str, neg_class: str) -> Callable[[pd.Series], pd.Series]:
        """Create pairwise labeling function."""

        def labeler(y: pd.Series) -> pd.Series:
            return y.map({pos_class: 1.0, neg_class: 0.0})

        return labeler

    @staticmethod
    def _make_composite_fn(pos_set: Set[str], neg_set: Set[str]) -> Callable[[pd.Series], pd.Series]:
        """Create composite labeling function."""

        def labeler(y: pd.Series) -> pd.Series:
            return y.map(
                lambda x: 1.0 if x in pos_set else (0.0 if x in neg_set else np.nan)
            )

        return labeler
