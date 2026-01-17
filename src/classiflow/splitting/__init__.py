"""Splitting utilities for cross-validation."""

from classiflow.splitting.group_stratified import (
    make_group_labels,
    iter_outer_splits,
    iter_inner_splits,
    assert_no_patient_leakage,
)

__all__ = [
    "make_group_labels",
    "iter_outer_splits",
    "iter_inner_splits",
    "assert_no_patient_leakage",
]
