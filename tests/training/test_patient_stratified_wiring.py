"""Integration-lite tests for patient-level stratification wiring."""

from pathlib import Path

import pandas as pd

from classiflow.config import TrainConfig, MulticlassConfig, MetaConfig
from classiflow.training.binary import train_binary_task
from classiflow.training.multiclass import train_multiclass_classifier
from classiflow.training.meta import train_meta_classifier


def _write_dataset(path: Path, n_patients: int = 4, rows_per_patient: int = 3) -> None:
    patients = [f"p{i}" for i in range(n_patients) for _ in range(rows_per_patient)]
    classes = ["A", "B", "C"]
    labels = []
    for i in range(n_patients):
        labels.extend([classes[i % len(classes)]] * rows_per_patient)
    df = pd.DataFrame(
        {
            "patient_id": patients,
            "label": labels[: len(patients)],
            "f1": list(range(len(patients))),
            "f2": list(range(len(patients))),
        }
    )
    df.to_csv(path, index=False)


def test_train_binary_patient_groups_passed(tmp_path, monkeypatch):
    data_path = tmp_path / "binary.csv"
    _write_dataset(data_path, n_patients=4, rows_per_patient=3)

    captured = {}

    def _stub_run_single_task(self, X, y, task_name, outdir, groups=None, patient_col=None):
        captured["groups"] = groups
        captured["patient_col"] = patient_col
        return {"inner_cv_rows": [], "inner_cv_split_rows": [], "outer_rows": [], "folds": []}

    monkeypatch.setattr("classiflow.training.binary.NestedCVOrchestrator.run_single_task", _stub_run_single_task)

    config = TrainConfig(
        data_path=data_path,
        label_col="label",
        pos_label="A",
        patient_col="patient_id",
        outdir=tmp_path / "out_binary",
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=7,
        smote_mode="off",
    )

    train_binary_task(config)

    assert captured["patient_col"] == "patient_id"
    assert captured["groups"] is not None


def test_train_multiclass_patient_groups_passed(tmp_path, monkeypatch):
    data_path = tmp_path / "multiclass.csv"
    _write_dataset(data_path, n_patients=4, rows_per_patient=3)

    captured = {}

    def _stub_run_multiclass_nested_cv(*args, **kwargs):
        captured["groups"] = kwargs.get("groups")
        return {"outdir": kwargs["config"].outdir, "n_folds": 2, "variants": ["none"]}

    monkeypatch.setattr("classiflow.training.multiclass._run_multiclass_nested_cv", _stub_run_multiclass_nested_cv)

    config = MulticlassConfig(
        data_path=data_path,
        label_col="label",
        classes=["A", "B", "C"],
        patient_col="patient_id",
        outdir=tmp_path / "out_multiclass",
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=7,
        smote_mode="off",
    )

    train_multiclass_classifier(config)

    assert captured["groups"] is not None


def test_train_meta_patient_groups_passed(tmp_path, monkeypatch):
    data_path = tmp_path / "meta.csv"
    _write_dataset(data_path, n_patients=4, rows_per_patient=3)

    captured = {}

    def _stub_run_meta_nested_cv(*args, **kwargs):
        captured["groups"] = kwargs.get("groups")
        return {"outdir": kwargs["config"].outdir, "n_tasks": 1, "n_folds": 2, "variants": ["none"]}

    monkeypatch.setattr("classiflow.training.meta._run_meta_nested_cv", _stub_run_meta_nested_cv)

    config = MetaConfig(
        data_path=data_path,
        label_col="label",
        classes=["A", "B", "C"],
        patient_col="patient_id",
        outdir=tmp_path / "out_meta",
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=7,
        smote_mode="off",
    )

    train_meta_classifier(config)

    assert captured["groups"] is not None
