"""Tests for Classiflow UI API."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

<<<<<<< HEAD
from classiflow.ui_api.app import create_app
from classiflow.ui_api.config import StorageMode, UIConfig
=======
from classiflow.ui_api.app import SPAStaticFiles, create_app, mount_static
from classiflow.ui_api.config import UIConfig, StorageMode
from classiflow.ui_api.scanner import LocalFilesystemScanner
from classiflow.ui_api.repositories.sqlite import SQLiteCommentReviewRepository
>>>>>>> origin/main
from classiflow.ui_api.models import ReviewStatus
from classiflow.ui_api.repositories.sqlite import SQLiteCommentReviewRepository
from classiflow.ui_api.scanner import LocalFilesystemScanner


@pytest.fixture
def temp_projects_dir():
    """Create a temporary projects directory with test data."""
    temp_dir = tempfile.mkdtemp()
    projects_root = Path(temp_dir) / "projects"
    projects_root.mkdir()

    # Create a test project
    project_dir = projects_root / "TEST_PROJECT__test_project"
    project_dir.mkdir()

    # Create project.yaml
    project_yaml = project_dir / "project.yaml"
    project_yaml.write_text(
        """
project:
  id: TEST_PROJECT
  name: Test Project
  description: A test project for unit testing
  owner: tester
data:
  train:
    manifest: /path/to/train.csv
  test:
    manifest: /path/to/test.csv
key_columns:
  label: target
task:
  mode: meta
execution:
  engine: torch
  device: mps
  model_set: torch_fast
models:
  candidates:
    - torch_mlp
  expanded_mlp_tuning_grid: true
  final_estimator_strategy: single
  technical_final_estimator_strategy: single
  bagging_n_estimators: 15
  bagging_max_samples: 0.8
  bagging_max_features: 0.6
  bagging_bootstrap: true
  bagging_bootstrap_features: false
validation:
  nested_cv:
    outer_folds: 3
    inner_folds: 5
    seed: 42
"""
    )

    # Create registry directory
    registry_dir = project_dir / "registry"
    registry_dir.mkdir()

    datasets_yaml = registry_dir / "datasets.yaml"
    datasets_yaml.write_text(
        """
datasets:
  train:
    dataset_type: train
    manifest_path: /path/to/train.csv
    sha256: abc123
    size_bytes: 1000
    registered_at: '2024-01-15T10:00:00'
    schema:
      columns: [feature1, feature2, target]
      feature_columns: [feature1, feature2]
    stats:
      rows: 100
      labels:
        A: 50
        B: 50
    dirty: false
updated_at: '2024-01-15T10:00:00'
"""
    )

    thresholds_yaml = registry_dir / "thresholds.yaml"
    thresholds_yaml.write_text("""
promotion_gate_template: clinical_conservative
""")

    # Create promotion directory
    promotion_dir = project_dir / "promotion"
    promotion_dir.mkdir()

    decision_yaml = promotion_dir / "decision.yaml"
    decision_yaml.write_text(
        """
decision: PASS
timestamp: '2024-01-15T12:00:00'
technical_run: run001
test_run: run002
reasons:
  - All thresholds met
override:
  enabled: false
"""
    )

    # Create runs directory
    runs_dir = project_dir / "runs"
    runs_dir.mkdir()

    # Create technical_validation phase
    tech_val_dir = runs_dir / "technical_validation" / "run001"
    tech_val_dir.mkdir(parents=True)

    # Create run.json
    run_json = tech_val_dir / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "run_id": "uuid-001",
                "timestamp": "2024-01-15T11:00:00",
                "package_version": "0.1.0",
                "training_data_path": "/path/to/train.csv",
                "training_data_hash": "abc123",
                "training_data_size_bytes": 1000,
                "training_data_row_count": 100,
                "config": {
                    "outer_folds": 3,
                    "label_col": "target",
                },
                "task_type": "meta",
                "python_version": "3.11.0",
                "feature_list": ["feature1", "feature2"],
            }
        )
    )

    # Create lineage.json
    lineage_json = tech_val_dir / "lineage.json"
    lineage_json.write_text(
        json.dumps(
            {
                "phase": "TECHNICAL_VALIDATION",
                "run_id": "run001",
                "timestamp_local": "2024-01-15T11:00:00",
                "classiflow_version": "0.1.0",
                "command": "classiflow project run-technical",
            }
        )
    )

    # Create metrics_summary.json
    metrics_json = tech_val_dir / "metrics_summary.json"
<<<<<<< HEAD
    metrics_json.write_text(
        json.dumps(
            {
                "summary": {
                    "balanced_accuracy": 0.85,
                    "f1_macro": 0.82,
                },
                "per_fold": {
                    "balanced_accuracy": [0.84, 0.85, 0.86],
                },
            }
        )
    )
=======
    metrics_json.write_text(json.dumps({
        "summary": {
            "balanced_accuracy": 0.85,
            "f1_macro": 0.82,
            "sensitivity": 0.86,
            "mcc": 0.65,
        },
        "per_fold": {
            "balanced_accuracy": [0.84, 0.85, 0.86],
        },
    }))
>>>>>>> origin/main

    # Create a test artifact (image)
    test_image = tech_val_dir / "roc_curve.png"
    # Create a minimal valid PNG (1x1 pixel)
    test_image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    # Create independent_test phase with bag-member summary
    test_run_dir = runs_dir / "independent_test" / "run002"
    (test_run_dir / "metrics").mkdir(parents=True)

    (test_run_dir / "run.json").write_text(json.dumps({
        "run_id": "uuid-002",
        "timestamp": "2024-01-15T12:00:00",
        "package_version": "0.1.0",
        "training_data_path": "/path/to/train.csv",
        "training_data_hash": "abc123",
        "training_data_row_count": 100,
        "config": {
            "label_col": "target",
            "models": {
                "final_estimator_strategy": "bagged",
            },
        },
        "task_type": "multiclass",
        "python_version": "3.11.0",
        "feature_list": ["feature1", "feature2"],
    }))

    (test_run_dir / "lineage.json").write_text(json.dumps({
        "phase": "INDEPENDENT_TEST",
        "run_id": "run002",
        "timestamp_local": "2024-01-15T12:00:00",
        "classiflow_version": "0.1.0",
        "command": "classiflow project run-test",
    }))

    (test_run_dir / "metrics.json").write_text(json.dumps({
        "overall": {
            "accuracy": 0.88,
            "balanced_accuracy": 0.87,
            "f1_macro": 0.86,
        },
    }))

    (test_run_dir / "bagging_summary.json").write_text(json.dumps({
        "strategy": "bagged",
        "member_count": 3,
        "estimator_type": "sklearn.linear_model.LogisticRegression",
        "evaluation_available": True,
        "metrics_csv_path": "metrics/bag_member_metrics.csv",
        "members": [
            {
                "member_index": 1,
                "estimator_type": "sklearn.linear_model.LogisticRegression",
                "accuracy": 0.84,
                "balanced_accuracy": 0.83,
                "f1_macro": 0.82,
                "mcc": 0.76,
                "roc_auc_macro": 0.91,
                "agreement_with_ensemble": 0.92,
            },
            {
                "member_index": 2,
                "estimator_type": "sklearn.linear_model.LogisticRegression",
                "accuracy": 0.86,
                "balanced_accuracy": 0.85,
                "f1_macro": 0.84,
                "mcc": 0.78,
                "roc_auc_macro": 0.92,
                "agreement_with_ensemble": 0.94,
            },
            {
                "member_index": 3,
                "estimator_type": "sklearn.linear_model.LogisticRegression",
                "accuracy": 0.82,
                "balanced_accuracy": 0.81,
                "f1_macro": 0.8,
                "mcc": 0.74,
                "roc_auc_macro": 0.9,
                "agreement_with_ensemble": 0.9,
            },
        ],
    }))

    (test_run_dir / "metrics" / "bag_member_metrics.csv").write_text(
        "member_index,accuracy,balanced_accuracy,f1_macro,mcc,roc_auc_macro,agreement_with_ensemble\n"
        "1,0.84,0.83,0.82,0.76,0.91,0.92\n"
        "2,0.86,0.85,0.84,0.78,0.92,0.94\n"
        "3,0.82,0.81,0.80,0.74,0.90,0.90\n"
    )

    # Create final_model phase with selected final-bundle summary
    final_run_dir = runs_dir / "final_model" / "run003"
    (final_run_dir / "registry").mkdir(parents=True)

    (final_run_dir / "run.json").write_text(json.dumps({
        "run_id": "uuid-003",
        "timestamp": "2024-01-15T13:00:00",
        "package_version": "0.1.0",
        "training_data_path": "/path/to/train.csv",
        "training_data_hash": "abc123",
        "training_data_row_count": 100,
        "config": {
            "task": {"mode": "multiclass"},
            "execution": {"engine": "torch", "device": "mps", "model_set": "torch_fast"},
            "models": {
                "selection_metric": "f1",
                "selection_direction": "max",
                "final_estimator_strategy": "bagged",
                "bagging_n_estimators": 15,
                "bagging_max_samples": 0.8,
            },
            "final_model": {
                "sampler": "none",
                "technical_run": str(tech_val_dir),
                "train_from_scratch": True,
            },
        },
        "task_type": "multiclass",
        "python_version": "3.11.0",
        "feature_list": ["feature1", "feature2"],
    }))

    (final_run_dir / "lineage.json").write_text(json.dumps({
        "phase": "FINAL_MODEL",
        "run_id": "run003",
        "timestamp_local": "2024-01-15T13:00:00",
        "classiflow_version": "0.1.0",
        "command": "classiflow project build-bundle",
    }))

    (final_run_dir / "registry" / "selected_binary_configs.json").write_text(json.dumps({
        "multiclass": {
            "task_name": "multiclass",
            "model_name": "torch_mlp",
            "sampler": "none",
            "mean_score": 0.91,
            "params": {"hidden_dim": 256, "n_layers": 3, "dropout": 0.2},
        },
    }))

    (final_run_dir / "final_model_summary.json").write_text(json.dumps({
        "run_id": "run003",
        "run_key": "TEST_PROJECT__test_project:final_model:run003",
        "task_type": "multiclass",
        "bundle_path": "model_bundle.zip",
        "technical_run": str(tech_val_dir),
        "sampler": "none",
        "train_from_scratch": True,
        "selection_metric": "f1",
        "selection_direction": "max",
        "execution": {"engine": "torch", "device": "mps", "model_set": "torch_fast"},
        "strategy": {
            "final_estimator_strategy": "bagged",
            "bagging_n_estimators": 15,
            "bagging_max_samples": 0.8,
        },
        "selected_models": [
            {
                "task_name": "multiclass",
                "model_name": "torch_mlp",
                "sampler": "none",
                "mean_score": 0.91,
                "params": {"hidden_dim": 256, "n_layers": 3, "dropout": 0.2},
            },
        ],
        "meta_model": None,
    }))

    (final_run_dir / "model_bundle.zip").write_bytes(b"bundle")

    yield projects_root

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_client(temp_projects_dir, temp_db_path):
    """Create a test client with temporary directories."""
    config = UIConfig(
        projects_root=temp_projects_dir,
        db_path=temp_db_path,
        storage_mode=StorageMode.LOCAL,
    )
    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_ok(self, test_client):
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["storage_mode"] == "local"
        assert data["project_count"] == 1

    def test_health_includes_version(self, test_client):
        response = test_client.get("/api/health")
        data = response.json()
        assert "version" in data


class TestProjectEndpoints:
    """Tests for project endpoints."""

    def test_list_projects(self, test_client):
        response = test_client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        project = data["items"][0]
        assert project["id"] == "TEST_PROJECT__test_project"
        assert project["name"] == "Test Project"
        assert project["gate_status"]["technical_validation"] == "PASS"
        assert project["gate_status"]["independent_test"] == "FAIL"

    def test_list_projects_with_search(self, test_client):
        response = test_client.get("/api/projects?q=test")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

        response = test_client.get("/api/projects?q=nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    def test_get_project(self, test_client):
        response = test_client.get("/api/projects/TEST_PROJECT__test_project")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "TEST_PROJECT__test_project"
        assert data["name"] == "Test Project"
        assert data["promotion"]["decision"] == "PASS"
        assert data["registry"]["thresholds"]["promotion_gate_template"] == "clinical_conservative"
        assert data["promotion"]["gates"]["technical_validation"]["passed"] is True
        assert data["promotion"]["gates"]["independent_test"]["passed"] is False
        assert data["model_settings"]["engine"] == "torch"
        assert data["model_settings"]["device"] == "mps"
        assert data["model_settings"]["model_set"] == "torch_fast"
        assert data["model_settings"]["expanded_mlp_tuning_grid"] is True
        assert data["model_settings"]["final_estimator_strategy"] == "single"
        assert data["model_settings"]["technical_final_estimator_strategy"] == "single"
        assert data["model_settings"]["bagging_n_estimators"] == 15
        assert data["selected_final_model"]["run_id"] == "run003"
        assert data["selected_final_model"]["selected_models"][0]["model_name"] == "torch_mlp"
        assert data["selected_final_model"]["strategy"]["bagging_max_samples"] == 0.8
        assert "technical_validation" in data["phases"]

    def test_get_project_not_found(self, test_client):
        response = test_client.get("/api/projects/nonexistent")
        assert response.status_code == 404

    def test_get_project_runs(self, test_client):
        response = test_client.get("/api/projects/TEST_PROJECT__test_project/runs")
        assert response.status_code == 200
        data = response.json()
        assert "technical_validation" in data
        assert len(data["technical_validation"]) == 1


class TestRunEndpoints:
    """Tests for run endpoints."""

    def test_get_run(self, test_client):
        run_key = "TEST_PROJECT__test_project:technical_validation:run001"
        response = test_client.get(f"/api/runs/{run_key}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "run001"
        assert data["phase"] == "technical_validation"
        assert data["task_type"] == "meta"
        assert data["metrics"]["primary"]["balanced_accuracy"] == 0.85

    def test_get_run_includes_bagging(self, test_client):
        run_key = "TEST_PROJECT__test_project:independent_test:run002"
        response = test_client.get(f"/api/runs/{run_key}")
        assert response.status_code == 200
        data = response.json()
        assert data["bagging"]["member_count"] == 3
        assert data["bagging"]["metrics_csv_path"] == "metrics/bag_member_metrics.csv"
        assert len(data["bagging"]["members"]) == 3

    def test_get_final_run_includes_selected_model(self, test_client):
        run_key = "TEST_PROJECT__test_project:final_model:run003"
        response = test_client.get(f"/api/runs/{run_key}")
        assert response.status_code == 200
        data = response.json()
        assert data["selected_final_model"]["run_id"] == "run003"
        assert data["selected_final_model"]["task_type"] == "multiclass"
        assert data["selected_final_model"]["selected_models"][0]["task_name"] == "multiclass"
        assert data["selected_final_model"]["selected_models"][0]["params"]["hidden_dim"] == 256

    def test_get_run_not_found(self, test_client):
        response = test_client.get("/api/runs/nonexistent:phase:run")
        assert response.status_code == 404

    def test_list_run_artifacts(self, test_client):
        run_key = "TEST_PROJECT__test_project:technical_validation:run001"
        response = test_client.get(f"/api/runs/{run_key}/artifacts")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        # Should find the PNG artifact
        artifacts = data["items"]
        png_artifacts = [a for a in artifacts if a["kind"] == "image"]
        assert len(png_artifacts) >= 1


class TestArtifactEndpoints:
    """Tests for artifact endpoints."""

    def test_get_artifact_by_path(self, test_client):
        response = test_client.get(
            "/api/projects/TEST_PROJECT__test_project/runs/technical_validation/run001/artifacts/roc_curve.png"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")

    def test_artifact_path_traversal_blocked(self, test_client):
        # Try path traversal
        response = test_client.get(
            "/api/projects/TEST_PROJECT__test_project/runs/technical_validation/run001/artifacts/../../../project.yaml"
        )
        assert response.status_code == 404

    def test_artifact_disallowed_extension(self, test_client, temp_projects_dir):
        # Create a disallowed file
        run_dir = temp_projects_dir / "TEST_PROJECT__test_project/runs/technical_validation/run001"
        bad_file = run_dir / "script.exe"
        bad_file.write_bytes(b"bad content")

        response = test_client.get(
            "/api/projects/TEST_PROJECT__test_project/runs/technical_validation/run001/artifacts/script.exe"
        )
        assert response.status_code == 404


class TestCommentEndpoints:
    """Tests for comment endpoints."""

    def test_create_comment(self, test_client):
        response = test_client.post(
            "/api/comments",
            json={
                "scope_type": "project",
                "scope_id": "TEST_PROJECT__test_project",
                "author": "tester",
                "content": "This is a test comment",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["author"] == "tester"
        assert data["content"] == "This is a test comment"
        assert "id" in data

    def test_list_comments(self, test_client):
        # Create a comment first
        test_client.post(
            "/api/comments",
            json={
                "scope_type": "project",
                "scope_id": "TEST_PROJECT__test_project",
                "author": "tester",
                "content": "Test comment",
            },
        )

        response = test_client.get(
            "/api/comments?scope_type=project&scope_id=TEST_PROJECT__test_project"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    def test_delete_comment(self, test_client):
        # Create a comment
        create_response = test_client.post(
            "/api/comments",
            json={
                "scope_type": "project",
                "scope_id": "TEST_PROJECT__test_project",
                "author": "tester",
                "content": "To be deleted",
            },
        )
        comment_id = create_response.json()["id"]

        # Delete it
        response = test_client.delete(f"/api/comments/{comment_id}")
        assert response.status_code == 200

        # Try to delete again
        response = test_client.delete(f"/api/comments/{comment_id}")
        assert response.status_code == 404


class TestReviewEndpoints:
    """Tests for review endpoints."""

    def test_create_review(self, test_client):
        response = test_client.post(
            "/api/reviews",
            json={
                "scope_type": "run",
                "scope_id": "TEST_PROJECT__test_project:technical_validation:run001",
                "reviewer": "reviewer1",
                "status": "approved",
                "notes": "Looks good!",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reviewer"] == "reviewer1"
        assert data["status"] == "approved"

    def test_update_review(self, test_client):
        # Create a review
        create_response = test_client.post(
            "/api/reviews",
            json={
                "scope_type": "run",
                "scope_id": "TEST_PROJECT__test_project:technical_validation:run001",
                "reviewer": "reviewer1",
                "status": "pending",
            },
        )
        review_id = create_response.json()["id"]

        # Update it
        response = test_client.patch(f"/api/reviews/{review_id}?status=approved&notes=Updated")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["notes"] == "Updated"


class TestStaticFrontendServing:
    """Tests for static frontend serving behavior."""

    def test_index_html_is_not_cached(self, tmp_path):
        static_dir = tmp_path / "dist"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<!doctype html><html><body>ui</body></html>")

        app = FastAPI()
        mount_static(app, static_dir)
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_spa_fallback_is_not_cached(self, tmp_path):
        static_dir = tmp_path / "dist"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<!doctype html><html><body>ui</body></html>")
        static_files = SPAStaticFiles(directory=static_dir, html=True)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/runs/some-run",
            "root_path": "",
            "scheme": "http",
            "headers": [(b"accept", b"text/html")],
            "query_string": b"",
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        response = anyio.run(static_files.get_response, "runs/some-run", scope)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_static_assets_keep_default_cache_headers(self, tmp_path):
        static_dir = tmp_path / "dist"
        assets_dir = static_dir / "assets"
        assets_dir.mkdir(parents=True)
        (static_dir / "index.html").write_text("<!doctype html><html><body>ui</body></html>")
        (assets_dir / "app.js").write_text("console.log('ui');")

        app = FastAPI()
        mount_static(app, static_dir)
        client = TestClient(app)

        response = client.get("/assets/app.js")
        assert response.status_code == 200
        assert response.headers.get("cache-control") != "no-store"


class TestScanner:
    """Tests for the filesystem scanner."""

    def test_scan_projects(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)
        projects = scanner.scan_projects()
        assert len(projects) == 1
        project = projects[0]
        assert project.id == "TEST_PROJECT__test_project"
        assert "technical_validation" in project.phases

    def test_get_run(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)
        scanner.scan_projects()
        run = scanner.get_run("TEST_PROJECT__test_project", "technical_validation", "run001")
        assert run is not None
        assert run.manifest.run_id == "run001"
        assert run.metrics["summary"]["balanced_accuracy"] == 0.85

    def test_get_run_refreshes_cache_when_metrics_change(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)
        scanner.scan_projects()
        run_key = ("TEST_PROJECT__test_project", "technical_validation", "run001")
        run = scanner.get_run(*run_key)
        assert run is not None
        assert run.metrics["summary"]["balanced_accuracy"] == 0.85

        metrics_path = (
            temp_projects_dir
            / "TEST_PROJECT__test_project"
            / "runs"
            / "technical_validation"
            / "run001"
            / "metrics_summary.json"
        )
        payload = json.loads(metrics_path.read_text())
        payload["summary"]["balanced_accuracy"] = 0.42
        metrics_path.write_text(json.dumps(payload))

        refreshed = scanner.get_run(*run_key)
        assert refreshed is not None
        assert refreshed.metrics["summary"]["balanced_accuracy"] == 0.42

    def test_compute_gate_results_uses_shared_promotion_evaluator(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)
        scanner.scan_projects()

        gates = scanner.compute_gate_results("TEST_PROJECT__test_project")

        assert gates["technical_validation"].passed is True
        assert gates["independent_test"].passed is False
        assert any(
            check.metric == "Sensitivity" and check.passed is False
            for check in gates["independent_test"].checks
        )

    def test_artifact_id_stability(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)
        run_key = "TEST_PROJECT__test_project:technical_validation:run001"

        # Get artifact ID twice
        id1 = scanner._generate_artifact_id(run_key, "roc_curve.png")
        id2 = scanner._generate_artifact_id(run_key, "roc_curve.png")

        # Should be identical
        assert id1 == id2

    def test_path_traversal_prevention(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)

        # Try path traversal
        path = scanner.resolve_artifact_path(
            "TEST_PROJECT__test_project",
            "technical_validation",
            "run001",
            "../../../project.yaml",
        )
        assert path is None

    def test_symlink_escape_prevention(self, temp_projects_dir):
        scanner = LocalFilesystemScanner(temp_projects_dir)

        # Create a symlink pointing outside
        run_dir = temp_projects_dir / "TEST_PROJECT__test_project/runs/technical_validation/run001"
        symlink = run_dir / "escape_link"
        try:
            symlink.symlink_to(temp_projects_dir / "..")
            path = scanner.resolve_artifact_path(
                "TEST_PROJECT__test_project",
                "technical_validation",
                "run001",
                "escape_link",
            )
            assert path is None
        finally:
            if symlink.exists():
                symlink.unlink()


class TestSQLiteRepository:
    """Tests for SQLite comment/review repository."""

    def test_create_and_list_comments(self, temp_db_path):
        repo = SQLiteCommentReviewRepository(temp_db_path)

        from classiflow.ui_api.models import CommentCreate

        # Create comments
        repo.create_comment(
            CommentCreate(
                scope_type="project",
                scope_id="test_project",
                author="user1",
                content="First comment",
            )
        )
        repo.create_comment(
            CommentCreate(
                scope_type="project",
                scope_id="test_project",
                author="user2",
                content="Second comment",
            )
        )

        # List
        comments, total = repo.list_comments("project", "test_project")
        assert total == 2
        assert len(comments) == 2

    def test_delete_comment(self, temp_db_path):
        repo = SQLiteCommentReviewRepository(temp_db_path)

        from classiflow.ui_api.models import CommentCreate

        comment = repo.create_comment(
            CommentCreate(
                scope_type="project",
                scope_id="test_project",
                author="user1",
                content="To delete",
            )
        )

        assert repo.delete_comment(comment.id)
        assert not repo.delete_comment(comment.id)  # Already deleted

    def test_review_status_update(self, temp_db_path):
        repo = SQLiteCommentReviewRepository(temp_db_path)

        from classiflow.ui_api.models import ReviewCreate

        review = repo.create_review(
            ReviewCreate(
                scope_type="run",
                scope_id="test_run",
                reviewer="reviewer1",
                status=ReviewStatus.pending,
            )
        )

        updated = repo.update_review_status(review.id, ReviewStatus.approved, "Looks good")
        assert updated is not None
        assert updated.status == ReviewStatus.approved
        assert updated.notes == "Looks good"
        assert updated.updated_at is not None
