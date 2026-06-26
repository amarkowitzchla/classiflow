"""Tests for project execution configuration wiring and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import typer

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
        expanded_mlp_tuning_grid=False,
        final_estimator_strategy="single",
        technical_final_estimator_strategy="single",
        bagging_n_estimators=10,
        bagging_max_samples=1.0,
        bagging_max_features=1.0,
        bagging_bootstrap=True,
        bagging_bootstrap_features=False,
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
    assert "tasks_only" not in config_data["task"]


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
        expanded_mlp_tuning_grid=False,
        final_estimator_strategy="single",
        technical_final_estimator_strategy="single",
        bagging_n_estimators=10,
        bagging_max_samples=1.0,
        bagging_max_features=1.0,
        bagging_bootstrap=True,
        bagging_bootstrap_features=False,
        test_id="TBIN",
    )

    root = project_root(tmp_path, "TBIN", "Torch Binary")
    config_data = load_yaml(root / "project.yaml")
    assert config_data["execution"]["engine"] == "torch"
    assert config_data["execution"]["device"] == "mps"
    assert config_data["execution"]["torch"]["dtype"] == "float32"
    assert config_data["execution"]["torch"]["num_workers"] == default_torch_num_workers()
    assert "multiclass" not in config_data


def test_project_bootstrap_accepts_calibration_options(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    bootstrap_project(
        train_manifest=train_manifest,
        test_manifest=None,
        name="Torch Multiclass",
        out_dir=tmp_path,
        mode="multiclass",
        engine="torch",
        device="cpu",
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
        calibration_enabled="true",
        calibration_method="temperature",
        expanded_mlp_tuning_grid=True,
        final_estimator_strategy="bagged",
        technical_final_estimator_strategy="single",
        bagging_n_estimators=5,
        bagging_max_samples=0.7,
        bagging_max_features=0.9,
        bagging_bootstrap=True,
        bagging_bootstrap_features=False,
        test_id="TMCAL",
    )

    root = project_root(tmp_path, "TMCAL", "Torch Multiclass")
    config_data = load_yaml(root / "project.yaml")
    assert config_data["calibration"]["enabled"] == "true"
    assert config_data["calibration"]["method"] == "temperature"
    assert config_data["models"]["expanded_mlp_tuning_grid"] is True
    assert config_data["models"]["final_estimator_strategy"] == "bagged"
    assert config_data["models"]["technical_final_estimator_strategy"] == "single"
    assert config_data["models"]["bagging_n_estimators"] == 5


def test_project_bootstrap_meta_persists_custom_tasks(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)
    tasks_json = tmp_path / "tasks.json"
    tasks_json.write_text('{"A_vs_B_only":{"pos":["A"],"neg":["B"]}}', encoding="utf-8")

    bootstrap_project(
        train_manifest=train_manifest,
        test_manifest=None,
        name="Meta Custom Tasks",
        out_dir=tmp_path,
        mode="meta",
        engine="sklearn",
        device=None,
        show_options=False,
        hierarchy=None,
        label_col="label",
        tasks_json=tasks_json,
        tasks_only=True,
        sample_id_col=None,
        patient_id_col=None,
        no_patient_stratified=False,
        thresholds=[],
        gate_profile="balanced",
        promotion_gate_template=None,
        list_promotion_gate_templates_flag=False,
        copy_data="pointer",
        test_id="TMETA",
    )

    root = project_root(tmp_path, "TMETA", "Meta Custom Tasks")
    config_data = load_yaml(root / "project.yaml")
    assert config_data["task"]["tasks_json"] == str(tasks_json.resolve())
    assert config_data["task"]["tasks_only"] is True


def test_project_bootstrap_tasks_only_requires_tasks_json(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    with pytest.raises(typer.BadParameter, match="--tasks-only requires --tasks-json"):
        bootstrap_project(
            train_manifest=train_manifest,
            test_manifest=None,
            name="Meta Missing Tasks",
            out_dir=tmp_path,
            mode="meta",
            engine="sklearn",
            device=None,
            show_options=False,
            hierarchy=None,
            label_col="label",
            tasks_json=None,
            tasks_only=True,
            sample_id_col=None,
            patient_id_col=None,
            no_patient_stratified=False,
            thresholds=[],
            gate_profile="balanced",
            promotion_gate_template=None,
            list_promotion_gate_templates_flag=False,
            copy_data="pointer",
            test_id="TMISS",
        )


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
    tasks_dir = tmp_path / "project" / "config"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks_json = tasks_dir / "tasks.json"
    tasks_json.write_text('{"A_vs_B_only":{"pos":["A"],"neg":["B"]}}', encoding="utf-8")

    config = ProjectConfig(
        project={"id": "B001", "name": "Backend"},
        data={"train": {"manifest": str(train_manifest)}},
        key_columns={"label": "label", "sample_id": "sample_id"},
        task={
            "mode": "meta",
            "patient_stratified": False,
            "tasks_json": "config/tasks.json",
            "tasks_only": True,
        },
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
        captured["tasks_json"] = train_config.tasks_json
        captured["tasks_only"] = train_config.tasks_only
        return {}

    monkeypatch.setattr(orchestrator, "train_meta_classifier", _fake_train_meta)
    monkeypatch.setattr(orchestrator, "_technical_metrics_from_run", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(orchestrator, "write_technical_report", lambda *_args, **_kwargs: None)

    orchestrator.run_technical_validation(paths, config)

    assert captured == {
        "backend": "torch",
        "device": "cpu",
        "model_set": "torch_fast",
        "tasks_json": tasks_json,
        "tasks_only": True,
    }


def test_project_run_technical_uses_technical_final_estimator_strategy(tmp_path: Path, monkeypatch) -> None:
    train_manifest = tmp_path / "train.csv"
    _write_manifest(train_manifest)

    config = ProjectConfig(
        project={"id": "B002", "name": "TechnicalStrategy"},
        data={"train": {"manifest": str(train_manifest)}},
        key_columns={"label": "label", "sample_id": "sample_id"},
        task={"mode": "binary", "patient_stratified": False},
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
        models={
            "final_estimator_strategy": "bagged",
            "technical_final_estimator_strategy": "single",
        },
    )

    paths = ProjectPaths(tmp_path / "project")
    paths.ensure()
    register_dataset(paths.datasets_yaml, config, "train", train_manifest)

    captured = {}

    def _fake_train_binary(train_config):
        captured["final_estimator_strategy"] = train_config.final_estimator_strategy
        captured["bagging_n_estimators"] = train_config.bagging_n_estimators
        return {}

    monkeypatch.setattr(orchestrator, "train_binary_task", _fake_train_binary)
    monkeypatch.setattr(orchestrator, "_technical_metrics_from_run", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(orchestrator, "write_technical_report", lambda *_args, **_kwargs: None)

    orchestrator.run_technical_validation(paths, config)

    assert captured == {
        "final_estimator_strategy": "single",
        "bagging_n_estimators": 10,
    }


def test_filter_model_params_maps_between_single_and_bagged_estimators():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression

    bagged = BaggingClassifier(estimator=LogisticRegression())
    single = LogisticRegression()

    assert orchestrator._filter_model_params(bagged, {"C": 2.0}) == {"estimator__C": 2.0}
    assert orchestrator._filter_model_params(single, {"estimator__C": 2.0}) == {"C": 2.0}


def test_effective_execution_payload_marks_hierarchical_runs_as_torch() -> None:
    config = ProjectConfig.model_validate(
        {
            "project": {"id": "H001", "name": "HierExecution"},
            "data": {"train": {"manifest": "train.csv"}},
            "key_columns": {"label": "Family_code"},
            "task": {
                "mode": "hierarchical",
                "patient_stratified": False,
                "hierarchy_path": "Class_v12_code",
            },
        }
    )

    execution = orchestrator._effective_execution_payload(config)

    assert execution["engine"] == "torch"
    assert execution["device"] == "auto"
    assert execution["model_set"] == "hierarchical_torch"
    assert isinstance(execution["torch"], dict)
    assert execution["torch"]["dtype"] == "float32"
    assert execution["torch"]["num_workers"] == default_torch_num_workers()
    assert execution["torch"]["require_device"] is False


def test_effective_execution_payload_preserves_non_hierarchical_execution() -> None:
    config = ProjectConfig.model_validate(
        {
            "project": {"id": "M001", "name": "MetaExecution"},
            "data": {"train": {"manifest": "train.csv"}},
            "key_columns": {"label": "label"},
            "task": {"mode": "meta", "patient_stratified": False},
            "execution": {
                "engine": "torch",
                "device": "cpu",
                "model_set": "torch_fast",
                "torch": {"dtype": "float16", "num_workers": 2, "require_device": True},
            },
        }
    )

    execution = orchestrator._effective_execution_payload(config)

    assert execution["engine"] == "torch"
    assert execution["device"] == "cpu"
    assert execution["model_set"] == "torch_fast"
    assert execution["torch"]["dtype"] == "float16"
    assert execution["torch"]["num_workers"] == 2
    assert execution["torch"]["require_device"] is True


def test_technical_hierarchical_metrics_from_run_extracts_l1_and_l2(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"fold": 1, "level": "L1", "accuracy": 0.9, "f1_macro": 0.88},
            {"fold": 2, "level": "L1", "accuracy": 0.8, "f1_macro": 0.78},
            {"fold": 1, "level": "L2_oracle_A", "accuracy": 0.7, "f1_macro": 0.68},
            {"fold": 2, "level": "L2_oracle_A", "accuracy": 0.9, "f1_macro": 0.89},
            {"fold": 1, "level": "L2_oracle_B", "accuracy": 0.6, "f1_macro": 0.58},
            {"fold": 2, "level": "pipeline", "accuracy": 0.75, "f1_macro": 0.72},
        ]
    ).to_csv(tmp_path / "metrics_outer_eval.csv", index=False)

    metrics = orchestrator._technical_hierarchical_metrics_from_run(tmp_path)

    assert metrics["L1"]["summary"]["accuracy"] == pytest.approx(0.85)
    assert metrics["L2"]["summary"]["accuracy"] == pytest.approx((0.7 + 0.9 + 0.6) / 3)
    assert metrics["L2_by_branch"]["A"]["summary"]["f1_macro"] == pytest.approx((0.68 + 0.89) / 2)
    assert metrics["pipeline"]["summary"]["accuracy"] == pytest.approx(0.75)
