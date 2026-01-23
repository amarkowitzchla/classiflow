from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from classiflow.cli.main import app


def test_cli_version_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "classiflow" in result.stdout


def test_cli_project_bootstrap_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    train_manifest = fixtures_dir / "tiny_train_manifest.csv"
    assert train_manifest.exists()

    projects_root = tmp_path / "projects"
    result = runner.invoke(
        app,
        [
            "project",
            "bootstrap",
            "--train-manifest",
            str(train_manifest),
            "--name",
            "CLI Smoke",
            "--out",
            str(projects_root),
            "--mode",
            "binary",
            "--copy-data",
            "pointer",
        ],
    )
    assert result.exit_code == 0, result.stdout

    project_dirs = [p for p in projects_root.iterdir() if p.is_dir()]
    assert len(project_dirs) == 1
    project_root = project_dirs[0]

    assert (project_root / "project.yaml").exists()
    assert (project_root / "registry" / "datasets.yaml").exists()
    assert (project_root / "registry" / "thresholds.yaml").exists()
    assert (project_root / "README.md").exists()

    datasets_text = (project_root / "registry" / "datasets.yaml").read_text(encoding="utf-8")
    assert "train:" in datasets_text
    assert "sha256:" in datasets_text
    assert "manifest_path:" in datasets_text
