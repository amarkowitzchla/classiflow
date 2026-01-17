"""Tests for group-aware stratified splitting utilities."""

import numpy as np
import pandas as pd

from classiflow.splitting import (
    iter_outer_splits,
    iter_inner_splits,
    assert_no_patient_leakage,
    make_group_labels,
)


def _make_group_df(n_patients: int = 6, rows_per_patient: int = 3) -> pd.DataFrame:
    patients = np.repeat([f"p{i}" for i in range(n_patients)], rows_per_patient)
    labels = np.repeat([0, 1, 0, 1, 0, 1], rows_per_patient)[: len(patients)]
    df = pd.DataFrame(
        {
            "patient_id": patients,
            "label": labels,
            "f1": np.arange(len(patients)),
        }
    )
    return df


def test_group_stratified_outer_and_inner_no_leakage():
    df = _make_group_df()
    y = df["label"]

    outer_splits = list(iter_outer_splits(df, y, "patient_id", n_splits=2, random_state=13))
    assert outer_splits, "Expected outer splits"

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_splits, 1):
        assert_no_patient_leakage(df, "patient_id", tr_idx, va_idx, f"outer fold {fold_idx}")

        df_tr = df.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        inner_splits = list(iter_inner_splits(df_tr, y_tr, "patient_id", n_splits=2, n_repeats=2, random_state=13))
        assert inner_splits, "Expected inner splits"
        for split_idx, (in_tr_idx, in_va_idx) in enumerate(inner_splits, 1):
            assert_no_patient_leakage(df_tr, "patient_id", in_tr_idx, in_va_idx, f"inner split {split_idx}")


def test_make_group_labels_conflict_raises():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "label": ["A", "B", "A"],
        }
    )
    try:
        make_group_labels(df, "patient_id", "label")
    except ValueError as exc:
        assert "p1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting patient labels")
