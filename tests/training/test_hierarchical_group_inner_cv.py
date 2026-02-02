import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import classiflow.training.hierarchical_cv as hier


def test_group_inner_cv_no_patient_overlap():
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "label": ["A", "A", "B", "B", "A", "A"],
    })
    splits = hier._build_group_inner_splits(
        df_tr=df,
        y_tr=df["label"].values,
        patient_col="patient_id",
        n_splits=2,
        n_repeats=1,
        random_state=0,
        context="test",
    )

    for tr_idx, va_idx in splits:
        tr_patients = set(df.iloc[tr_idx]["patient_id"])
        va_patients = set(df.iloc[va_idx]["patient_id"])
        assert tr_patients.isdisjoint(va_patients)


def test_group_inner_cv_conflicting_labels_raises():
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p3"],
        "label": ["A", "B", "A", "B"],
    })
    with pytest.raises(ValueError, match="Patient label conflict"):
        hier._build_group_inner_splits(
            df_tr=df,
            y_tr=df["label"].values,
            patient_col="patient_id",
            n_splits=2,
            n_repeats=1,
            random_state=0,
            context="test",
        )
