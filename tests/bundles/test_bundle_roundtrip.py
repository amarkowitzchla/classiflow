"""Tests for bundle create/inspect/load roundtrip."""

import pytest
import json
import joblib
from pathlib import Path
import tempfile
import zipfile

from classiflow.bundles import create_bundle, inspect_bundle, load_bundle
from classiflow.lineage import create_training_manifest


@pytest.fixture
def mock_run_dir(tmp_path):
    """Create a mock training run directory."""
    run_dir = tmp_path / "mock_run"
    run_dir.mkdir()

    # Create run manifest
    manifest = create_training_manifest(
        data_path=Path("data.csv"),
        data_hash="a" * 64,
        data_size_bytes=1024,
        data_row_count=100,
        config={"outer_folds": 3},
        task_type="binary",
        feature_list=["feat1", "feat2"],
    )
    manifest.save(run_dir / "run.json")

    # Create fold directory
    fold1_dir = run_dir / "fold1"
    fold1_dir.mkdir()

    # Create mock model artifacts
    binary_dir = fold1_dir / "binary_smote"
    binary_dir.mkdir()

    mock_model = {"type": "mock_model", "params": {}}
    joblib.dump(mock_model, binary_dir / "binary_pipes.joblib")
    joblib.dump(mock_model, binary_dir / "meta_model.joblib")

    # Create mock metrics
    with open(run_dir / "metrics_inner_cv.csv", "w") as f:
        f.write("task,model,accuracy\n")
        f.write("task1,lr,0.85\n")

    return run_dir


def test_create_bundle(mock_run_dir, tmp_path):
    """Test creating a bundle."""
    out_bundle = tmp_path / "test_bundle.zip"

    bundle_path = create_bundle(
        run_dir=mock_run_dir,
        out_bundle=out_bundle,
        fold=1,
        include_all_folds=False,
        include_metrics=True,
    )

    assert bundle_path.exists()
    assert bundle_path.suffix == ".zip"
    assert zipfile.is_zipfile(bundle_path)


def test_inspect_bundle(mock_run_dir, tmp_path):
    """Test inspecting a bundle."""
    out_bundle = tmp_path / "test_bundle.zip"

    create_bundle(
        run_dir=mock_run_dir,
        out_bundle=out_bundle,
        fold=1,
    )

    info = inspect_bundle(out_bundle)

    assert info["valid"] is True
    assert "run_manifest" in info
    assert "artifacts" in info
    assert "version" in info
    assert info["file_count"] > 0


def test_load_bundle(mock_run_dir, tmp_path):
    """Test loading a bundle."""
    out_bundle = tmp_path / "test_bundle.zip"

    create_bundle(
        run_dir=mock_run_dir,
        out_bundle=out_bundle,
        fold=1,
    )

    bundle_data = load_bundle(out_bundle, fold=1)

    assert "loader" in bundle_data
    assert "manifest" in bundle_data
    assert "fold_dir" in bundle_data

    # Check manifest
    manifest = bundle_data["manifest"]
    assert manifest.run_id is not None
    assert manifest.task_type == "binary"


def test_bundle_roundtrip(mock_run_dir, tmp_path):
    """Test full roundtrip: create → inspect → load."""
    out_bundle = tmp_path / "roundtrip_bundle.zip"

    # Create
    bundle_path = create_bundle(
        run_dir=mock_run_dir,
        out_bundle=out_bundle,
        fold=1,
    )

    # Inspect
    info = inspect_bundle(bundle_path)
    assert info["valid"]

    # Load
    bundle_data = load_bundle(bundle_path, fold=1)
    manifest = bundle_data["manifest"]

    # Verify run_id matches
    assert manifest.run_id == info["run_manifest"]["run_id"]

    # Cleanup loader
    bundle_data["loader"].cleanup()


def test_bundle_includes_required_files(mock_run_dir, tmp_path):
    """Test that bundle includes all required files."""
    out_bundle = tmp_path / "complete_bundle.zip"

    create_bundle(
        run_dir=mock_run_dir,
        out_bundle=out_bundle,
        fold=1,
        include_metrics=True,
    )

    info = inspect_bundle(out_bundle)

    required_files = ["run.json", "artifacts.json", "version.txt", "README.txt"]
    for required in required_files:
        assert required in info["file_list"], f"Missing required file: {required}"
