"""Unit tests for project module core logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from classiflow.lineage.hashing import compute_file_hash
from classiflow.projects.dataset_registry import register_dataset, verify_manifest_hash
from classiflow.projects.project_models import ProjectConfig
from classiflow.projects.promotion import evaluate_promotion
from classiflow.projects.project_models import ThresholdsConfig


def _write_manifest(path: Path, include_patient: bool = True) -> None:
    df = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "label": ["A", "B", "A", "B"],
        "feat1": [1.0, 2.0, 3.0, 4.0],
    })
    if include_patient:
        df["patient_id"] = ["p1", "p2", "p3", "p4"]
    df.to_csv(path, index=False)


def _project_config(train_path: Path, test_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project={"id": "T001", "name": "Test"},
        data={
            "train": {"manifest": str(train_path)},
            "test": {"manifest": str(test_path)},
        },
    )


def test_register_dataset_hash_matches(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_manifest(train_path)
    _write_manifest(test_path)

    config = _project_config(train_path, test_path)
    registry_path = tmp_path / "datasets.yaml"

    entry = register_dataset(registry_path, config, "train", train_path)
    assert entry.sha256 == compute_file_hash(train_path)


def test_register_dataset_missing_columns_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.csv"
    pd.DataFrame({"sample_id": ["s1"], "feat": [1]}).to_csv(manifest, index=False)
    config = _project_config(manifest, manifest)
    registry_path = tmp_path / "datasets.yaml"

    with pytest.raises(ValueError, match="Missing required columns"):
        register_dataset(registry_path, config, "train", manifest)


def test_verify_manifest_hash_does_not_parse_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "data.csv"
    _write_manifest(manifest)
    expected = compute_file_hash(manifest)

    def _raise(*_args, **_kwargs):
        raise AssertionError("pandas.read_csv should not be called")

    monkeypatch.setattr(pd, "read_csv", _raise)
    assert verify_manifest_hash(manifest, expected)


def test_promotion_gate_evaluation_missing_metrics() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"accuracy": 0.9}
    thresholds.independent_test.required = {"accuracy": 0.9}

    results = evaluate_promotion(thresholds, {}, {}, {})
    assert not results["technical_validation"].passed
    assert not results["independent_test"].passed


def test_promotion_fails_on_poor_calibration() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"recall": 0.8}
    thresholds.independent_test.required = {"recall": 0.8}
    thresholds.promotion.calibration.brier_max = 0.2
    thresholds.promotion.calibration.ece_max = 0.25

    tech_metrics = {"recall": 0.9, "brier_calibrated": 0.3, "ece_calibrated": 0.1}
    test_metrics = {"recall": 0.92, "brier_calibrated": 0.3, "ece_calibrated": 0.2}

    results = evaluate_promotion(thresholds, tech_metrics, {}, test_metrics)
    assert not results["technical_validation"].passed
    assert not results["independent_test"].passed


def test_promotion_ignores_calibration_when_not_configured() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"recall": 0.8}
    thresholds.independent_test.required = {"recall": 0.8}

    tech_metrics = {"recall": 0.9}
    test_metrics = {"recall": 0.92}

    results = evaluate_promotion(thresholds, tech_metrics, {}, test_metrics)
    assert results["technical_validation"].passed
    assert results["independent_test"].passed


def test_promotion_enforces_only_configured_calibration_metric() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"recall": 0.8}
    thresholds.independent_test.required = {"recall": 0.8}
    thresholds.promotion.calibration.brier_max = 0.2
    thresholds.promotion.calibration.ece_max = None

    tech_metrics = {"recall": 0.9, "brier_calibrated": 0.3}
    test_metrics = {"recall": 0.92, "brier_calibrated": 0.12}

    results = evaluate_promotion(thresholds, tech_metrics, {}, test_metrics)
    assert not results["technical_validation"].passed
    assert results["independent_test"].passed


def test_promotion_resolves_f1_to_weighted_when_macro_missing() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"f1": 0.7}
    thresholds.independent_test.required = {"f1": 0.7}

    tech_metrics = {"f1_weighted": 0.8}
    test_metrics = {"f1_weighted": 0.85}

    results = evaluate_promotion(thresholds, tech_metrics, {}, test_metrics)
    assert results["technical_validation"].passed
    assert results["independent_test"].passed


def test_promotion_passes_when_decision_and_calibration_pass() -> None:
    thresholds = ThresholdsConfig()
    thresholds.technical_validation.required = {"recall": 0.8}
    thresholds.independent_test.required = {"recall": 0.8}
    thresholds.promotion.calibration.brier_max = 0.2
    thresholds.promotion.calibration.ece_max = 0.25

    tech_metrics = {"recall": 0.9, "brier_calibrated": 0.1, "ece_calibrated": 0.1}
    test_metrics = {"recall": 0.92, "brier_calibrated": 0.12, "ece_calibrated": 0.2}

    results = evaluate_promotion(thresholds, tech_metrics, {}, test_metrics)
    assert results["technical_validation"].passed
    assert results["independent_test"].passed
