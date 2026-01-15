"""Tests for feature preprocessing and alignment."""

import pytest
import pandas as pd
import numpy as np

from classiflow.inference.preprocess import (
    FeatureAligner,
    validate_input_data,
    compute_feature_stats,
)


class TestFeatureAligner:
    """Test FeatureAligner class."""

    def test_align_perfect_match(self):
        """Test alignment when all features match."""
        required_features = ["a", "b", "c"]
        aligner = FeatureAligner(required_features, strict=True)

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "extra": [10, 11, 12],  # Extra column should be dropped
        })

        X, metadata, warnings = aligner.align(df)

        assert list(X.columns) == required_features
        assert len(X) == 3
        assert len(warnings) == 0

    def test_align_strict_mode_missing_features(self):
        """Test strict mode fails on missing features."""
        required_features = ["a", "b", "c"]
        aligner = FeatureAligner(required_features, strict=True)

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            # Missing 'c'
        })

        with pytest.raises(ValueError, match="Strict mode"):
            aligner.align(df)

    def test_align_lenient_mode_fills_missing(self):
        """Test lenient mode fills missing features."""
        required_features = ["a", "b", "c"]
        aligner = FeatureAligner(required_features, strict=False, fill_strategy="zero")

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            # Missing 'c'
        })

        X, metadata, warnings = aligner.align(df)

        assert list(X.columns) == required_features
        assert (X["c"] == 0).all()
        assert len(warnings) > 0  # Should warn about missing features

    def test_align_lenient_median_fill(self):
        """Test lenient mode with median fill."""
        required_features = ["a", "b", "c"]
        training_stats = {
            "c": {"median": 5.0}
        }
        aligner = FeatureAligner(
            required_features,
            strict=False,
            fill_strategy="median",
            training_stats=training_stats,
        )

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            # Missing 'c'
        })

        X, metadata, warnings = aligner.align(df)

        assert (X["c"] == 5.0).all()

    def test_align_preserves_metadata(self):
        """Test metadata columns are preserved."""
        required_features = ["a", "b"]
        aligner = FeatureAligner(required_features, strict=True)

        df = pd.DataFrame({
            "id": ["s1", "s2", "s3"],
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "label": ["A", "B", "A"],
        })

        X, metadata, warnings = aligner.align(df, id_col="id", label_col="label")

        assert "id" in metadata.columns
        assert "label" in metadata.columns
        assert list(X.columns) == required_features

    def test_align_type_coercion(self):
        """Test type coercion to numeric."""
        required_features = ["a", "b"]
        aligner = FeatureAligner(required_features, strict=True)

        df = pd.DataFrame({
            "a": ["1", "2", "3"],  # String numbers
            "b": [4, 5, 6],
        })

        X, metadata, warnings = aligner.align(df)

        assert pd.api.types.is_numeric_dtype(X["a"])

    def test_align_handles_nan(self):
        """Test NaN handling."""
        required_features = ["a", "b"]
        aligner = FeatureAligner(required_features, strict=True)

        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [4, 5, 6],
        })

        X, metadata, warnings = aligner.align(df)

        # NaN should be filled with 0
        assert not X["a"].isna().any()
        assert len(warnings) > 0  # Should warn about NaN


class TestValidateInputData:
    """Test validate_input_data function."""

    def test_empty_dataframe(self):
        """Test validation of empty dataframe."""
        df = pd.DataFrame()
        warnings = validate_input_data(df)

        assert len(warnings) > 0
        assert any("empty" in w.lower() for w in warnings)

    def test_duplicate_ids(self):
        """Test detection of duplicate IDs."""
        df = pd.DataFrame({
            "id": ["a", "b", "a"],  # Duplicate 'a'
            "x": [1, 2, 3],
        })

        warnings = validate_input_data(df, id_col="id")

        assert any("duplicate" in w.lower() for w in warnings)

    def test_all_null_columns(self):
        """Test detection of all-null columns."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "null_col": [np.nan, np.nan, np.nan],
        })

        warnings = validate_input_data(df)

        assert any("null" in w.lower() for w in warnings)

    def test_infinite_values(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            "x": [1, np.inf, 3],
        })

        warnings = validate_input_data(df)

        assert any("infinite" in w.lower() for w in warnings)


class TestComputeFeatureStats:
    """Test compute_feature_stats function."""

    def test_compute_stats(self):
        """Test computation of feature statistics."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        stats = compute_feature_stats(df)

        assert "a" in stats
        assert "b" in stats

        assert stats["a"]["mean"] == 3.0
        assert stats["a"]["median"] == 3.0
        assert stats["b"]["mean"] == 30.0

    def test_handles_nan(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
        })

        stats = compute_feature_stats(df)

        # Should compute stats excluding NaN
        assert stats["a"]["mean"] == 2.0
        assert stats["a"]["median"] == 2.0
