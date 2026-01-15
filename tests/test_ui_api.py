"""Tests for Classiflow UI API."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from classiflow.ui_api.app import create_app
from classiflow.ui_api.config import UIConfig, StorageMode
from classiflow.ui_api.scanner import LocalFilesystemScanner
from classiflow.ui_api.repositories.sqlite import SQLiteCommentReviewRepository
from classiflow.ui_api.models import ReviewStatus


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
    project_yaml.write_text("""
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
validation:
  nested_cv:
    outer_folds: 3
    inner_folds: 5
    seed: 42
""")

    # Create registry directory
    registry_dir = project_dir / "registry"
    registry_dir.mkdir()

    datasets_yaml = registry_dir / "datasets.yaml"
    datasets_yaml.write_text("""
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
""")

    # Create promotion directory
    promotion_dir = project_dir / "promotion"
    promotion_dir.mkdir()

    decision_yaml = promotion_dir / "decision.yaml"
    decision_yaml.write_text("""
decision: PASS
timestamp: '2024-01-15T12:00:00'
technical_run: run001
test_run: run002
reasons:
  - All thresholds met
override:
  enabled: false
""")

    # Create runs directory
    runs_dir = project_dir / "runs"
    runs_dir.mkdir()

    # Create technical_validation phase
    tech_val_dir = runs_dir / "technical_validation" / "run001"
    tech_val_dir.mkdir(parents=True)

    # Create run.json
    run_json = tech_val_dir / "run.json"
    run_json.write_text(json.dumps({
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
    }))

    # Create lineage.json
    lineage_json = tech_val_dir / "lineage.json"
    lineage_json.write_text(json.dumps({
        "phase": "TECHNICAL_VALIDATION",
        "run_id": "run001",
        "timestamp_local": "2024-01-15T11:00:00",
        "classiflow_version": "0.1.0",
        "command": "classiflow project run-technical",
    }))

    # Create metrics_summary.json
    metrics_json = tech_val_dir / "metrics_summary.json"
    metrics_json.write_text(json.dumps({
        "summary": {
            "balanced_accuracy": 0.85,
            "f1_macro": 0.82,
        },
        "per_fold": {
            "balanced_accuracy": [0.84, 0.85, 0.86],
        },
    }))

    # Create a test artifact (image)
    test_image = tech_val_dir / "roc_curve.png"
    # Create a minimal valid PNG (1x1 pixel)
    test_image.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
        b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )

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
        response = test_client.patch(
            f"/api/reviews/{review_id}?status=approved&notes=Updated"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["notes"] == "Updated"


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
        comment1 = repo.create_comment(CommentCreate(
            scope_type="project",
            scope_id="test_project",
            author="user1",
            content="First comment",
        ))
        comment2 = repo.create_comment(CommentCreate(
            scope_type="project",
            scope_id="test_project",
            author="user2",
            content="Second comment",
        ))

        # List
        comments, total = repo.list_comments("project", "test_project")
        assert total == 2
        assert len(comments) == 2

    def test_delete_comment(self, temp_db_path):
        repo = SQLiteCommentReviewRepository(temp_db_path)

        from classiflow.ui_api.models import CommentCreate

        comment = repo.create_comment(CommentCreate(
            scope_type="project",
            scope_id="test_project",
            author="user1",
            content="To delete",
        ))

        assert repo.delete_comment(comment.id)
        assert not repo.delete_comment(comment.id)  # Already deleted

    def test_review_status_update(self, temp_db_path):
        repo = SQLiteCommentReviewRepository(temp_db_path)

        from classiflow.ui_api.models import ReviewCreate

        review = repo.create_review(ReviewCreate(
            scope_type="run",
            scope_id="test_run",
            reviewer="reviewer1",
            status=ReviewStatus.pending,
        ))

        updated = repo.update_review_status(review.id, ReviewStatus.approved, "Looks good")
        assert updated is not None
        assert updated.status == ReviewStatus.approved
        assert updated.notes == "Looks good"
        assert updated.updated_at is not None
