"""Tests for task builder."""

import pytest
import pandas as pd
import numpy as np

from classiflow.tasks import TaskBuilder


def test_task_builder_ovr():
    """Test OvR task generation."""
    classes = ["A", "B", "C"]
    builder = TaskBuilder(classes).build_ovr_tasks()
    tasks = builder.get_tasks()

    assert len(tasks) == 3
    assert "A_vs_Rest" in tasks
    assert "B_vs_Rest" in tasks
    assert "C_vs_Rest" in tasks

    # Test labeling function
    y = pd.Series(["A", "B", "C", "A", "B"])
    y_bin = tasks["A_vs_Rest"](y)

    assert list(y_bin) == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_task_builder_pairwise():
    """Test pairwise task generation."""
    classes = ["A", "B", "C"]
    builder = TaskBuilder(classes).build_pairwise_tasks()
    tasks = builder.get_tasks()

    assert len(tasks) == 3  # C(3,2) = 3
    assert "A_vs_B" in tasks
    assert "A_vs_C" in tasks
    assert "B_vs_C" in tasks

    # Test labeling
    y = pd.Series(["A", "B", "C", "A", "B"])
    y_bin = tasks["A_vs_B"](y)

    expected = [1.0, 0.0, np.nan, 1.0, 0.0]
    assert pd.isna(y_bin[2])
    assert list(y_bin[[0, 1, 3, 4]]) == [1.0, 0.0, 1.0, 0.0]


def test_task_builder_composite():
    """Test composite task."""
    classes = ["A", "B", "C", "D"]
    builder = TaskBuilder(classes).add_composite_task(
        name="AB_vs_CD",
        pos_classes=["A", "B"],
        neg_classes=["C", "D"],
    )
    tasks = builder.get_tasks()

    assert "AB_vs_CD" in tasks

    y = pd.Series(["A", "B", "C", "D", "A"])
    y_bin = tasks["AB_vs_CD"](y)

    assert list(y_bin) == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_task_builder_composite_rest():
    """Test composite task with 'rest' negative."""
    classes = ["A", "B", "C"]
    builder = TaskBuilder(classes).add_composite_task(
        name="A_vs_Rest",
        pos_classes=["A"],
        neg_classes="rest",
    )
    tasks = builder.get_tasks()

    y = pd.Series(["A", "B", "C", "A"])
    y_bin = tasks["A_vs_Rest"](y)

    assert list(y_bin) == [1.0, 0.0, 0.0, 1.0]


def test_task_builder_all():
    """Test building all auto tasks."""
    classes = ["A", "B", "C"]
    builder = TaskBuilder(classes).build_all_auto_tasks()
    tasks = builder.get_tasks()

    # 3 OvR + 3 pairwise = 6
    assert len(tasks) == 6
