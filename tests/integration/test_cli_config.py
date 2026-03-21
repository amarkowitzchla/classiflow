from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from classiflow.cli.main import app


def test_project_bootstrap_show_options_lists_enums() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["project", "bootstrap", "--show-options"])
    assert result.exit_code == 0
    assert "modes:" in result.stdout
    assert "engines:" in result.stdout
    assert "sklearn, torch, hybrid" in result.stdout


def test_config_show_emits_execution_block() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["config", "show", "--mode", "binary", "--engine", "torch", "--device", "mps"],
    )
    assert result.exit_code == 0
    assert "execution:" in result.stdout
    assert "engine: torch" in result.stdout
    assert "device: mps" in result.stdout
    assert "torch:" in result.stdout
    assert "expanded_mlp_tuning_grid: false" in result.stdout
    assert "final_estimator_strategy: single" in result.stdout


def test_config_validate_reports_actionable_errors(tmp_path: Path) -> None:
    bad_config = tmp_path / "project.yaml"
    bad_config.write_text(
        """
project:
  id: P001
  name: Bad

data:
  train:
    manifest: train.csv

execution:
  engine: sklearn
  torch:
    dtype: float32
    num_workers: 0
    require_device: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["config", "validate", str(bad_config)])
    assert result.exit_code == 1
    assert "execution.torch is not allowed" in result.output


def test_config_validate_rejects_bagging_for_meta_mode(tmp_path: Path) -> None:
    bad_config = tmp_path / "project.yaml"
    bad_config.write_text(
        """
project:
  id: P002
  name: BadMetaBagging

data:
  train:
    manifest: train.csv

task:
  mode: meta
  patient_stratified: false

models:
  final_estimator_strategy: bagged
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["config", "validate", str(bad_config)])
    assert result.exit_code == 1
    assert "final_estimator_strategy=bagged is not supported for task.mode=meta" in result.output
