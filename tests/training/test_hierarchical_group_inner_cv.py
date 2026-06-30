import pandas as pd
import pytest

pytest.importorskip("torch")
import classiflow.training.hierarchical_cv as hier
from classiflow.plots import hierarchical as hplots


def test_group_inner_cv_no_patient_overlap():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "label": ["A", "A", "B", "B", "A", "A"],
        }
    )
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
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "label": ["A", "B", "A", "B"],
        }
    )
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


def test_curve_data_guard_raises_probability_shape_mismatch():
    y_true = np.array([0, 1])
    y_proba = np.full((2, 3), 1 / 3)

    with pytest.raises(ValueError, match="class count mismatch"):
        hier._compute_roc_pr_curve_data(
            y_true,
            y_proba,
            n_classes=2,
            context="test fold",
        )


def test_curve_data_uses_fallback_when_sklearn_curve_raises(monkeypatch):
    def _raise_index_error(*_args, **_kwargs):
        raise IndexError("index 33700 is out of bounds for axis 0 with size 33700")

    monkeypatch.setattr(hplots, "roc_curve", _raise_index_error)
    monkeypatch.setattr(hplots, "precision_recall_curve", _raise_index_error)
    monkeypatch.setattr(hplots, "average_precision_score", _raise_index_error)

    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=int)
    y_proba = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.20, 0.70],
            [0.10, 0.75, 0.15],
            [0.70, 0.20, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=float,
    )

    result = hier._compute_roc_pr_curve_data(
        y_true,
        y_proba,
        n_classes=3,
        context="fallback test fold",
    )

    assert result is not None


def test_curve_data_returns_none_when_binary_fold_has_single_class():
    y_true = np.ones(6, dtype=int)
    y_proba = np.array(
        [
            [0.05, 0.95],
            [0.02, 0.98],
            [0.10, 0.90],
            [0.08, 0.92],
            [0.03, 0.97],
            [0.07, 0.93],
        ],
        dtype=float,
    )

    result = hier._compute_roc_pr_curve_data(
        y_true,
        y_proba,
        n_classes=2,
        context="single-class binary fold",
    )

    assert result is None
