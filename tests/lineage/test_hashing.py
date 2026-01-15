"""Tests for data hashing utilities."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from classiflow.lineage.hashing import (
    compute_file_hash,
    compute_dataframe_hash,
    compute_canonical_hash,
    get_file_metadata,
)


def test_compute_file_hash():
    """Test file hashing is stable and reproducible."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test content\n")
        temp_path = Path(f.name)

    try:
        # Hash should be reproducible
        hash1 = compute_file_hash(temp_path)
        hash2 = compute_file_hash(temp_path)
        assert hash1 == hash2

        # Hash should be hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length

        # Different content should give different hash
        with open(temp_path, "w") as f:
            f.write("different content\n")

        hash3 = compute_file_hash(temp_path)
        assert hash3 != hash1

    finally:
        temp_path.unlink()


def test_compute_dataframe_hash_canonical():
    """Test dataframe hashing with canonical ordering."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3]})  # Different column order

    # With canonical=True, hashes should match
    hash1 = compute_dataframe_hash(df1, canonical=True)
    hash2 = compute_dataframe_hash(df2, canonical=True)

    assert hash1 == hash2


def test_compute_dataframe_hash_non_canonical():
    """Test dataframe hashing without canonical ordering."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3]})  # Different column order

    # Without canonical, hashes may differ
    # (depends on internal representation, but generally should differ)
    hash1 = compute_dataframe_hash(df1, canonical=False)
    hash2 = compute_dataframe_hash(df2, canonical=False)

    # Both should be valid hashes
    assert isinstance(hash1, str)
    assert isinstance(hash2, str)
    assert len(hash1) == 64
    assert len(hash2) == 64


def test_compute_canonical_hash_numpy():
    """Test canonical hashing of numpy arrays."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

    hash1 = compute_canonical_hash(arr)
    hash2 = compute_canonical_hash(arr)

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64


def test_compute_canonical_hash_bytes():
    """Test canonical hashing of bytes."""
    data = b"test data"

    hash1 = compute_canonical_hash(data)
    hash2 = compute_canonical_hash(data)

    assert hash1 == hash2


def test_get_file_metadata():
    """Test file metadata extraction."""
    # Create temp CSV
    df = pd.DataFrame({"A": range(10), "B": range(10, 20)})

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        temp_path = Path(f.name)

    try:
        df.to_csv(temp_path, index=False)

        metadata = get_file_metadata(temp_path)

        assert "size_bytes" in metadata
        assert "row_count" in metadata
        assert "sha256_hash" in metadata

        assert metadata["size_bytes"] > 0
        assert metadata["row_count"] == 10
        assert len(metadata["sha256_hash"]) == 64

    finally:
        temp_path.unlink()


def test_hash_stability_across_runs():
    """Test that hashes are stable across multiple runs."""
    df = pd.DataFrame({
        "feature1": np.random.RandomState(42).randn(100),
        "feature2": np.random.RandomState(43).randn(100),
    })

    hashes = [compute_dataframe_hash(df, canonical=True) for _ in range(5)]

    # All hashes should be identical
    assert len(set(hashes)) == 1
