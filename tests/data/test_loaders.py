"""Tests for classiflow.data.loaders module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from classiflow.data import DataSpec, DataFormat, LoadedDataset, load_table, load_data, infer_format


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def iris_like_df():
    """Create an iris-like synthetic dataset."""
    np.random.seed(42)
    n_samples = 150
    n_features = 4

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels (3 classes, 50 each)
    labels = np.array(["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50)
    np.random.shuffle(labels)

    # Create DataFrame
    df = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    df["species"] = labels
    df["sample_id"] = [f"sample_{i:03d}" for i in range(n_samples)]
    df["batch"] = np.random.choice(["batch_A", "batch_B"], size=n_samples)

    return df


@pytest.fixture
def temp_csv_file(tmp_path, iris_like_df):
    """Write iris-like data to a CSV file."""
    csv_path = tmp_path / "test_data.csv"
    iris_like_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_parquet_file(tmp_path, iris_like_df):
    """Write iris-like data to a Parquet file."""
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "test_data.parquet"
    iris_like_df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def temp_parquet_dataset(tmp_path, iris_like_df):
    """Create a directory with chunked parquet files."""
    pytest.importorskip("pyarrow")
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # Split data into 3 chunks
    chunk_size = 50
    for i in range(3):
        chunk_df = iris_like_df.iloc[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = dataset_dir / f"part-{i:03d}.parquet"
        chunk_df.to_parquet(chunk_path, index=False)

    return dataset_dir


# ============================================================================
# Test DataFormat
# ============================================================================


class TestDataFormat:
    """Tests for DataFormat enum."""

    def test_from_path_csv(self, tmp_path):
        """Test CSV format detection."""
        csv_path = tmp_path / "data.csv"
        csv_path.touch()
        assert DataFormat.from_path(csv_path) == DataFormat.CSV

    def test_from_path_parquet(self, tmp_path):
        """Test Parquet format detection."""
        parquet_path = tmp_path / "data.parquet"
        parquet_path.touch()
        assert DataFormat.from_path(parquet_path) == DataFormat.PARQUET

    def test_from_path_directory(self, tmp_path):
        """Test directory format detection (parquet dataset)."""
        assert DataFormat.from_path(tmp_path) == DataFormat.PARQUET_DATASET

    def test_from_path_unknown(self, tmp_path):
        """Test error on unknown format."""
        unknown_path = tmp_path / "data.xyz"
        unknown_path.touch()
        with pytest.raises(ValueError, match="Cannot infer data format"):
            DataFormat.from_path(unknown_path)


# ============================================================================
# Test infer_format
# ============================================================================


class TestInferFormat:
    """Tests for infer_format function."""

    def test_infer_csv(self, temp_csv_file):
        """Test CSV format inference."""
        assert infer_format(temp_csv_file) == DataFormat.CSV

    def test_infer_parquet(self, temp_parquet_file):
        """Test Parquet format inference."""
        assert infer_format(temp_parquet_file) == DataFormat.PARQUET

    def test_infer_dataset_dir(self, temp_parquet_dataset):
        """Test dataset directory format inference."""
        assert infer_format(temp_parquet_dataset) == DataFormat.PARQUET_DATASET


# ============================================================================
# Test load_table
# ============================================================================


class TestLoadTable:
    """Tests for load_table function."""

    def test_load_csv(self, temp_csv_file, iris_like_df):
        """Test loading CSV file."""
        df = load_table(temp_csv_file)
        assert len(df) == len(iris_like_df)
        assert list(df.columns) == list(iris_like_df.columns)

    def test_load_parquet(self, temp_parquet_file, iris_like_df):
        """Test loading Parquet file."""
        df = load_table(temp_parquet_file)
        assert len(df) == len(iris_like_df)
        assert set(df.columns) == set(iris_like_df.columns)

    def test_load_parquet_dataset(self, temp_parquet_dataset, iris_like_df):
        """Test loading Parquet dataset directory."""
        df = load_table(temp_parquet_dataset)
        assert len(df) == len(iris_like_df)
        assert set(df.columns) == set(iris_like_df.columns)

    def test_load_csv_with_columns(self, temp_csv_file):
        """Test loading CSV with column selection."""
        df = load_table(temp_csv_file, columns=["sepal_length", "species"])
        assert list(df.columns) == ["sepal_length", "species"]

    def test_load_parquet_with_columns(self, temp_parquet_file):
        """Test loading Parquet with column selection."""
        df = load_table(temp_parquet_file, columns=["sepal_length", "species"])
        assert list(df.columns) == ["sepal_length", "species"]


# ============================================================================
# Test DataSpec
# ============================================================================


class TestDataSpec:
    """Tests for DataSpec dataclass."""

    def test_format_inference(self, temp_csv_file):
        """Test automatic format inference."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        assert spec.format == DataFormat.CSV

    def test_to_dict(self, temp_csv_file):
        """Test conversion to dictionary."""
        spec = DataSpec(
            path=temp_csv_file,
            label_col="species",
            id_col="sample_id",
            classes=["setosa", "versicolor"],
        )
        d = spec.to_dict()
        assert d["path"] == str(temp_csv_file)
        assert d["label_col"] == "species"
        assert d["id_col"] == "sample_id"
        assert d["classes"] == ["setosa", "versicolor"]

    def test_from_dict(self, temp_csv_file):
        """Test creation from dictionary."""
        d = {
            "path": str(temp_csv_file),
            "format": "csv",
            "label_col": "species",
        }
        spec = DataSpec.from_dict(d)
        assert spec.path == temp_csv_file
        assert spec.format == DataFormat.CSV
        assert spec.label_col == "species"


# ============================================================================
# Test load_data
# ============================================================================


class TestLoadData:
    """Tests for load_data function."""

    def test_load_csv_basic(self, temp_csv_file, iris_like_df):
        """Test basic CSV loading with labels."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        assert isinstance(dataset, LoadedDataset)
        assert dataset.n_samples == len(iris_like_df)
        assert dataset.n_features == 4  # numeric columns only
        assert dataset.n_classes == 3
        assert set(dataset.class_names) == {"setosa", "versicolor", "virginica"}

    def test_load_parquet_basic(self, temp_parquet_file, iris_like_df):
        """Test basic Parquet loading with labels."""
        spec = DataSpec(path=temp_parquet_file, label_col="species")
        dataset = load_data(spec)

        assert dataset.n_samples == len(iris_like_df)
        assert dataset.n_features == 4
        assert dataset.n_classes == 3

    def test_load_dataset_directory(self, temp_parquet_dataset, iris_like_df):
        """Test loading from parquet dataset directory."""
        spec = DataSpec(path=temp_parquet_dataset, label_col="species")
        dataset = load_data(spec)

        assert dataset.n_samples == len(iris_like_df)
        assert dataset.n_classes == 3

    def test_feature_auto_detection(self, temp_csv_file):
        """Test automatic numeric feature detection."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        # Should detect only numeric columns (excluding label)
        expected_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        assert set(dataset.feature_names) == set(expected_features)

    def test_explicit_feature_cols(self, temp_csv_file):
        """Test explicit feature column specification."""
        spec = DataSpec(
            path=temp_csv_file,
            label_col="species",
            feature_cols=["sepal_length", "sepal_width"],
        )
        dataset = load_data(spec)

        assert dataset.n_features == 2
        assert set(dataset.feature_names) == {"sepal_length", "sepal_width"}

    def test_id_column_extraction(self, temp_csv_file):
        """Test ID column extraction."""
        spec = DataSpec(path=temp_csv_file, label_col="species", id_col="sample_id")
        dataset = load_data(spec)

        assert dataset.ids is not None
        assert len(dataset.ids) == dataset.n_samples
        assert "sample_000" in dataset.ids

    def test_group_column_extraction(self, temp_csv_file):
        """Test group column extraction."""
        spec = DataSpec(path=temp_csv_file, label_col="species", group_col="batch")
        dataset = load_data(spec)

        assert dataset.groups is not None
        assert len(dataset.groups) == dataset.n_samples
        assert set(dataset.groups) == {"batch_A", "batch_B"}

    def test_class_filtering(self, temp_csv_file):
        """Test filtering to subset of classes."""
        spec = DataSpec(
            path=temp_csv_file, label_col="species", classes=["setosa", "versicolor"]
        )
        dataset = load_data(spec)

        assert dataset.n_classes == 2
        assert set(dataset.class_names) == {"setosa", "versicolor"}

    def test_label_not_in_features(self, temp_csv_file):
        """Test that label column is not included in features."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        assert "species" not in dataset.feature_names

    def test_id_not_in_features(self, temp_csv_file):
        """Test that ID column is not included in features."""
        spec = DataSpec(path=temp_csv_file, label_col="species", id_col="sample_id")
        dataset = load_data(spec)

        assert "sample_id" not in dataset.feature_names

    def test_group_not_in_features(self, temp_csv_file):
        """Test that group column is not included in features."""
        spec = DataSpec(path=temp_csv_file, label_col="species", group_col="batch")
        dataset = load_data(spec)

        assert "batch" not in dataset.feature_names

    def test_deterministic_feature_order(self, temp_csv_file):
        """Test that feature order is deterministic."""
        spec = DataSpec(path=temp_csv_file, label_col="species")

        # Load multiple times
        dataset1 = load_data(spec)
        dataset2 = load_data(spec)

        assert dataset1.feature_names == dataset2.feature_names

    def test_metadata_dataframe(self, temp_csv_file):
        """Test that df_meta contains metadata columns."""
        spec = DataSpec(
            path=temp_csv_file, label_col="species", id_col="sample_id", group_col="batch"
        )
        dataset = load_data(spec)

        assert dataset.df_meta is not None
        assert "sample_id" in dataset.df_meta.columns
        assert "species" in dataset.df_meta.columns
        assert "batch" in dataset.df_meta.columns

    def test_x_is_float32(self, temp_csv_file):
        """Test that X array is float32 for memory efficiency."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        assert dataset.X.dtype == np.float32

    def test_missing_label_column_error(self, temp_csv_file):
        """Test error when label column doesn't exist."""
        spec = DataSpec(path=temp_csv_file, label_col="nonexistent")

        with pytest.raises(ValueError, match="Label column"):
            load_data(spec)

    def test_missing_id_column_error(self, temp_csv_file):
        """Test error when ID column doesn't exist."""
        spec = DataSpec(path=temp_csv_file, label_col="species", id_col="nonexistent")

        with pytest.raises(ValueError, match="ID column"):
            load_data(spec)

    def test_file_not_found_error(self, tmp_path):
        """Test error when file doesn't exist."""
        spec = DataSpec(path=tmp_path / "nonexistent.csv", label_col="species")

        with pytest.raises(FileNotFoundError):
            load_data(spec)


# ============================================================================
# Test LoadedDataset
# ============================================================================


class TestLoadedDataset:
    """Tests for LoadedDataset dataclass."""

    def test_properties(self, temp_csv_file):
        """Test dataset properties."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        assert dataset.n_samples == 150
        assert dataset.n_features == 4
        assert dataset.n_classes == 3
        assert len(dataset.class_names) == 3

    def test_class_counts(self, temp_csv_file):
        """Test class_counts property."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        counts = dataset.class_counts
        assert sum(counts.values()) == dataset.n_samples

    def test_to_dataframe(self, temp_csv_file):
        """Test to_dataframe method."""
        spec = DataSpec(path=temp_csv_file, label_col="species", id_col="sample_id")
        dataset = load_data(spec)

        df = dataset.to_dataframe(include_meta=True)
        assert "_id" in df.columns
        assert "_label" in df.columns
        assert len(df) == dataset.n_samples

    def test_get_X_df(self, temp_csv_file):
        """Test get_X_df method."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        X_df = dataset.get_X_df()
        assert isinstance(X_df, pd.DataFrame)
        assert list(X_df.columns) == dataset.feature_names
        assert len(X_df) == dataset.n_samples

    def test_get_y_series(self, temp_csv_file):
        """Test get_y_series method."""
        spec = DataSpec(path=temp_csv_file, label_col="species")
        dataset = load_data(spec)

        y_series = dataset.get_y_series()
        assert isinstance(y_series, pd.Series)
        assert len(y_series) == dataset.n_samples

    def test_subset(self, temp_csv_file):
        """Test subset method."""
        spec = DataSpec(path=temp_csv_file, label_col="species", id_col="sample_id")
        dataset = load_data(spec)

        indices = np.array([0, 10, 20, 30, 40])
        subset = dataset.subset(indices)

        assert subset.n_samples == 5
        assert subset.n_features == dataset.n_features
        assert len(subset.ids) == 5

    def test_validation_x_y_mismatch(self):
        """Test validation error for X and y length mismatch."""
        with pytest.raises(ValueError, match="X and y length mismatch"):
            LoadedDataset(
                X=np.random.randn(10, 5).astype(np.float32),
                y=np.array(["A"] * 8),  # Wrong length
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )

    def test_validation_feature_names_mismatch(self):
        """Test validation error for feature_names length mismatch."""
        with pytest.raises(ValueError, match="feature_names length"):
            LoadedDataset(
                X=np.random.randn(10, 5).astype(np.float32),
                y=np.array(["A"] * 10),
                feature_names=["f1", "f2", "f3"],  # Wrong length
            )


# ============================================================================
# Test CLI Data Path Resolution (integration)
# ============================================================================


class TestCLIDataPathResolution:
    """Tests for CLI --data vs --data-csv resolution."""

    def test_resolve_data_path_prefers_data(self, temp_csv_file, temp_parquet_file):
        """Test that --data takes precedence over --data-csv."""
        from classiflow.config import _resolve_data_path

        resolved = _resolve_data_path(data=temp_parquet_file, data_csv=temp_csv_file)
        assert resolved == temp_parquet_file

    def test_resolve_data_path_falls_back_to_csv(self, temp_csv_file):
        """Test that --data-csv is used when --data is None."""
        from classiflow.config import _resolve_data_path

        resolved = _resolve_data_path(data=None, data_csv=temp_csv_file)
        assert resolved == temp_csv_file

    def test_resolve_data_path_error_when_both_none(self):
        """Test error when neither --data nor --data-csv provided."""
        from classiflow.config import _resolve_data_path

        with pytest.raises(ValueError, match="Either --data or --data-csv must be provided"):
            _resolve_data_path(data=None, data_csv=None)
