"""Tests for manifest management."""

import pytest
import json
from pathlib import Path
import tempfile

from classiflow.lineage.manifest import (
    TrainingRunManifest,
    InferenceRunManifest,
    create_training_manifest,
    create_inference_manifest,
    validate_manifest_compatibility,
)


def test_training_manifest_creation():
    """Test creating a training run manifest."""
    manifest = create_training_manifest(
        data_path=Path("data.csv"),
        data_hash="abc123" * 10 + "abcd",
        data_size_bytes=1024,
        data_row_count=100,
        config={"outer_folds": 3, "random_state": 42},
        task_type="binary",
        feature_list=["feature1", "feature2", "feature3"],
    )

    assert manifest.run_id is not None
    assert len(manifest.run_id) == 36  # UUID4 format
    assert manifest.training_data_hash == "abc123" * 10 + "abcd"
    assert manifest.training_data_row_count == 100
    assert manifest.task_type == "binary"
    assert len(manifest.feature_list) == 3


def test_training_manifest_save_load():
    """Test saving and loading training manifest."""
    manifest = create_training_manifest(
        data_path=Path("data.csv"),
        data_hash="def456" * 10 + "efgh",
        data_size_bytes=2048,
        data_row_count=200,
        config={"test": "config"},
        task_type="meta",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "run.json"
        manifest.save(save_path)

        # Load and verify
        loaded = TrainingRunManifest.load(save_path)

        assert loaded.run_id == manifest.run_id
        assert loaded.training_data_hash == manifest.training_data_hash
        assert loaded.training_data_row_count == manifest.training_data_row_count
        assert loaded.task_type == manifest.task_type


def test_inference_manifest_creation():
    """Test creating an inference run manifest."""
    parent_run_id = "12345678-1234-1234-1234-123456789012"

    manifest = create_inference_manifest(
        parent_run_id=parent_run_id,
        data_path=Path("test_data.csv"),
        data_hash="xyz789" * 10 + "wxyz",
        data_size_bytes=512,
        data_row_count=50,
        config={"strict_features": True},
    )

    assert manifest.inference_run_id is not None
    assert len(manifest.inference_run_id) == 36
    assert manifest.parent_run_id == parent_run_id
    assert manifest.inference_data_hash == "xyz789" * 10 + "wxyz"
    assert manifest.inference_data_row_count == 50


def test_inference_manifest_save_load():
    """Test saving and loading inference manifest."""
    manifest = create_inference_manifest(
        parent_run_id="parent-id-12345",
        data_path=Path("test.csv"),
        data_hash="a" * 64,
        data_size_bytes=1024,
        data_row_count=100,
        config={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "inference_run.json"
        manifest.save(save_path)

        # Load and verify
        loaded = InferenceRunManifest.load(save_path)

        assert loaded.inference_run_id == manifest.inference_run_id
        assert loaded.parent_run_id == manifest.parent_run_id


def test_validate_manifest_compatibility_exact_match():
    """Test validation with exact feature match."""
    train_manifest = create_training_manifest(
        data_path=Path("train.csv"),
        data_hash="b" * 64,
        data_size_bytes=1024,
        data_row_count=100,
        config={},
        feature_list=["feature1", "feature2", "feature3"],
    )

    inference_features = ["feature1", "feature2", "feature3"]

    compatible, warnings = validate_manifest_compatibility(
        train_manifest,
        inference_features,
    )

    assert compatible is True
    assert len(warnings) == 0


def test_validate_manifest_compatibility_missing_features():
    """Test validation with missing features."""
    train_manifest = create_training_manifest(
        data_path=Path("train.csv"),
        data_hash="c" * 64,
        data_size_bytes=1024,
        data_row_count=100,
        config={},
        feature_list=["feature1", "feature2", "feature3"],
    )

    inference_features = ["feature1", "feature2"]  # Missing feature3

    compatible, warnings = validate_manifest_compatibility(
        train_manifest,
        inference_features,
    )

    assert compatible is False
    assert len(warnings) > 0
    assert any("Missing" in w for w in warnings)


def test_validate_manifest_compatibility_extra_features():
    """Test validation with extra features."""
    train_manifest = create_training_manifest(
        data_path=Path("train.csv"),
        data_hash="d" * 64,
        data_size_bytes=1024,
        data_row_count=100,
        config={},
        feature_list=["feature1", "feature2"],
    )

    inference_features = ["feature1", "feature2", "feature3", "feature4"]

    compatible, warnings = validate_manifest_compatibility(
        train_manifest,
        inference_features,
    )

    assert compatible is True  # Extra features OK
    assert len(warnings) > 0
    assert any("Extra" in w for w in warnings)
