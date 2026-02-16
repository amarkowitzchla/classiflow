"""Tests for project execution configuration wiring and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from classiflow.cli.project import bootstrap_project
from classiflow.config import default_torch_num_workers
from classiflow.projects import orchestrator
from classiflow.projects.dataset_registry import register_dataset
from classiflow.projects.project_fs import ProjectPaths, choose_project_id, project_root
from classiflow.projects.project_models import ProjectConfig
from classiflow.projects.yaml_utils import load_yaml


def _write_manifest(path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "label": ["A", "B", "A", "B"],
            "feat1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(path, index=False)


def test_project_bootstrap_sklearn_is_minimal(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)
    name = "Backend Project"
    test_id = "T_BACKEND"

    bootstrap_project(
        train_manifest=train_manifest,
        test_manifest=None,
        name=name,
        out_dir=tmp_path,
        mode="multiclass",
        engine="sklearn",
        device=None,
        show_options=False,
        hierarchy=None,
        label_col="label",
        sample_id_col=None,
        patient_id_col=None,
        no_patient_stratified=False,
        thresholds=[],
        gate_profile="balanced",
        promotion_gate_template=None,
        list_promotion_gate_templates_flag=False,
        copy_data="pointer",
        test_id=test_id,
    )

    project_id = choose_project_id(name, test_id)
    root = project_root(tmp_path, project_id, name)
    config_data = load_yaml(root / "project.yaml")

    assert "backend" not in config_data
    assert "torch_dtype" not in config_data
    assert "device" not in config_data
    assert config_data["execution"] == {"engine": "sklearn"}
    assert config_data["multiclass"]["backend"] == "sklearn_cpu"
    assert "torch" not in config_data["execution"]


def test_project_bootstrap_torch_binary_excludes_multiclass(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    bootstrap_project(
        train_manifest=train_manifest,
        test_manifest=None,
        name="Torch Binary",
        out_dir=tmp_path,
        mode="binary",
        engine="torch",
        device="mps",
        show_options=False,
        hierarchy=None,
        label_col="label",
        sample_id_col=None,
        patient_id_col=None,
        no_patient_stratified=False,
        thresholds=[],
        gate_profile="balanced",
        promotion_gate_template=None,
        list_promotion_gate_templates_flag=False,
        copy_data="pointer",
        test_id="TBIN",
    )

    root = project_root(tmp_path, "TBIN", "Torch Binary")
    config_data = load_yaml(root / "project.yaml")
    assert config_data["execution"]["engine"] == "torch"
    assert config_data["execution"]["device"] == "mps"
    assert config_data["execution"]["torch"]["dtype"] == "float32"
    assert config_data["execution"]["torch"]["num_workers"] == default_torch_num_workers()
    assert "multiclass" not in config_data


def test_legacy_backend_fields_are_normalized() -> None:
    config = ProjectConfig.model_validate(
        {
            "project": {"id": "B001", "name": "Backend"},
            "data": {"train": {"manifest": "train.csv"}},
            "key_columns": {"label": "label", "sample_id": "sample_id"},
            "task": {"mode": "meta", "patient_stratified": False},
            "backend": "torch",
            "device": "cpu",
            "model_set": "torch_fast",
            "torch_dtype": "float32",
            "torch_num_workers": 0,
        }
    )

    assert config.execution.engine == "torch"
    assert config.execution.device == "cpu"
    assert config.execution.model_set == "torch_fast"
    assert config.torch_dtype == "float32"
    assert config.torch_num_workers == 0


def test_legacy_calibration_toggle_is_normalized() -> None:
    config = ProjectConfig.model_validate(
        {
            "project": {"id": "C001", "name": "Calibration"},
            "data": {"train": {"manifest": "train.csv"}},
            "key_columns": {"label": "label"},
            "task": {"mode": "meta", "patient_stratified": False},
            "calibration": {"calibrate_meta": False, "method": "sigmoid"},
        }
    )

    assert config.calibration.enabled == "false"
    assert config.calibration.method == "sigmoid"


def test_project_config_allows_temperature_calibration_method() -> None:
    config = ProjectConfig.model_validate(
        {
            "project": {"id": "C002", "name": "CalibrationTemp"},
            "data": {"train": {"manifest": "train.csv"}},
            "task": {"mode": "multiclass", "patient_stratified": False},
            "execution": {
                "engine": "torch",
                "device": "cpu",
                "torch": {"dtype": "float32", "num_workers": 0, "require_device": False},
            },
            "calibration": {"enabled": "true", "method": "temperature"},
        }
    )

    assert config.calibration.enabled == "true"
    assert config.calibration.method == "temperature"


def test_sklearn_engine_rejects_torch_subtree() -> None:
    with pytest.raises(ValueError, match="execution.torch is not allowed"):
        ProjectConfig.model_validate(
            {
                "project": {"id": "B001", "name": "Backend"},
                "data": {"train": {"manifest": "train.csv"}},
                "task": {"mode": "meta", "patient_stratified": False},
                "execution": {
                    "engine": "sklearn",
                    "torch": {"dtype": "float32", "num_workers": 0, "require_device": False},
                },
            }
        )


def test_project_run_passes_backend_settings(tmp_path: Path, monkeypatch) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    config = ProjectConfig(
        project={"id": "B001", "name": "Backend"},
        data={"train": {"manifest": str(train_manifest)}},
        key_columns={"label": "label", "sample_id": "sample_id"},
        task={"mode": "meta", "patient_stratified": False},
        execution={
            "engine": "torch",
            "device": "cpu",
            "model_set": "torch_fast",
            "torch": {
                "dtype": "float32",
                "num_workers": 0,
                "require_device": False,
            },
        },
    )

    paths = ProjectPaths(tmp_path / "project")
    paths.ensure()
    register_dataset(paths.datasets_yaml, config, "train", train_manifest)

    captured = {}

    def _fake_train_meta(train_config):
        captured["backend"] = train_config.backend
        captured["device"] = train_config.device
        captured["model_set"] = train_config.model_set
        return {}

    monkeypatch.setattr(orchestrator, "train_meta_classifier", _fake_train_meta)
    monkeypatch.setattr(orchestrator, "_technical_metrics_from_run", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(orchestrator, "write_technical_report", lambda *_args, **_kwargs: None)

    orchestrator.run_technical_validation(paths, config)

    assert captured == {
        "backend": "torch",
        "device": "cpu",
        "model_set": "torch_fast",
    }
