"""Tests for project backend configuration wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classiflow.cli.project import bootstrap_project
from classiflow.projects.project_fs import ProjectPaths, choose_project_id, project_root
from classiflow.projects.project_models import ProjectConfig
from classiflow.projects.dataset_registry import register_dataset
from classiflow.projects import orchestrator
from classiflow.projects.yaml_utils import load_yaml


def _write_manifest(path: Path) -> None:
    df = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "label": ["A", "B", "A", "B"],
        "feat1": [1.0, 2.0, 3.0, 4.0],
    })
    df.to_csv(path, index=False)


def test_project_bootstrap_writes_backend_fields(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)
    name = "Backend Project"
    test_id = "T_BACKEND"

    bootstrap_project(
        train_manifest=train_manifest,
        test_manifest=None,
        name=name,
        out_dir=tmp_path,
        mode="meta",
        hierarchy=None,
        label_col="label",
        sample_id_col=None,
        patient_id_col=None,
        no_patient_stratified=False,
        thresholds=[],
        copy_data="pointer",
        test_id=test_id,
    )

    project_id = choose_project_id(name, test_id)
    root = project_root(tmp_path, project_id, name)
    config_data = load_yaml(root / "project.yaml")
    for key in ["backend", "device", "model_set", "torch_dtype", "torch_num_workers"]:
        assert key in config_data


def test_project_run_passes_backend_settings(tmp_path: Path, monkeypatch) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    config = ProjectConfig(
        project={"id": "B001", "name": "Backend"},
        data={"train": {"manifest": str(train_manifest)}},
        key_columns={"label": "label", "sample_id": "sample_id"},
        task={"mode": "meta", "patient_stratified": False},
        backend="torch",
        device="cpu",
        model_set="torch_fast",
        torch_dtype="float32",
        torch_num_workers=0,
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
