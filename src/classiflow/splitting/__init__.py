"""Splitting utilities for cross-validation."""

from classiflow.splitting.group_stratified import (
    assert_no_patient_leakage,
    iter_inner_splits,
    iter_outer_splits,
    make_group_labels,
)

__all__ = [
    "make_group_labels",
    "iter_outer_splits",
    "iter_inner_splits",
    "assert_no_patient_leakage",
]
