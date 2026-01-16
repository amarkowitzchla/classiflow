"""Data loading functions for Classiflow.

This module provides unified data loading for CSV, Parquet, and Parquet dataset
directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd

from classiflow.data.spec import DataSpec, DataFormat
from classiflow.data.dataset import LoadedDataset
from classiflow.data.validation import validate_columns, validate_features

logger = logging.getLogger(__name__)

# Flag to track PyArrow availability
_PYARROW_AVAILABLE: Optional[bool] = None


def validate_parquet_available() -> None:
    """
    Check if PyArrow is available for Parquet operations.

    Raises
    ------
    ImportError
        If PyArrow is not installed with helpful installation message
    """
    global _PYARROW_AVAILABLE

    if _PYARROW_AVAILABLE is None:
        try:
            import pyarrow  # noqa: F401

            _PYARROW_AVAILABLE = True
        except ImportError:
            _PYARROW_AVAILABLE = False

    if not _PYARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is required for Parquet support but is not installed.\n"
            "Install with: pip install classiflow[parquet] or pip install pyarrow"
        )


def infer_format(path: Path) -> DataFormat:
    """
    Infer data format from file path.

    Parameters
    ----------
    path : Path
        Path to data file or directory

    Returns
    -------
    DataFormat
        Inferred format (csv, parquet, or parquet_dataset)

    Raises
    ------
    ValueError
        If format cannot be inferred
    """
    return DataFormat.from_path(path)


def load_table(
    path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load a table from file (CSV, Parquet, or Parquet dataset directory).

    This is a simple loader that returns a raw DataFrame without any
    feature/label processing. Use load_data() for ML-ready datasets.

    Parameters
    ----------
    path : Path
        Path to data file or directory
    columns : List[str], optional
        Subset of columns to load (performance optimization)
    filters : dict, optional
        Filters for parquet dataset (partition filters)

    Returns
    -------
    pd.DataFrame
        Loaded data

    Examples
    --------
    >>> df = load_table(Path("data.parquet"))
    >>> df = load_table(Path("data.csv"))
    >>> df = load_table(Path("data_parquet_dataset/"))
    """
    path = Path(path)
    fmt = infer_format(path)

    logger.info(f"Loading table from {path} (format: {fmt.value})")

    if fmt == DataFormat.CSV:
        return _load_csv(path, columns=columns)
    elif fmt == DataFormat.PARQUET:
        validate_parquet_available()
        return _load_parquet(path, columns=columns)
    elif fmt == DataFormat.PARQUET_DATASET:
        validate_parquet_available()
        return _load_parquet_dataset(path, columns=columns, filters=filters)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_data(spec: DataSpec) -> LoadedDataset:
    """
    Load and prepare a dataset for ML training/inference.

    This is the main entry point for loading data. It:
    1. Loads data from CSV, Parquet, or Parquet dataset directory
    2. Validates required columns exist
    3. Extracts features (auto-detects numeric if not specified)
    4. Extracts labels, IDs, and groups
    5. Ensures deterministic column ordering
    6. Validates feature quality

    Parameters
    ----------
    spec : DataSpec
        Data specification containing path, column names, etc.

    Returns
    -------
    LoadedDataset
        Container with X, y, feature_names, ids, groups, and df_meta

    Raises
    ------
    ValueError
        If required columns are missing or data is invalid
    FileNotFoundError
        If data file/directory does not exist

    Examples
    --------
    >>> spec = DataSpec(
    ...     path=Path("data.parquet"),
    ...     label_col="subtype",
    ...     group_col="patient_id",
    ... )
    >>> dataset = load_data(spec)
    >>> print(f"Loaded {dataset.n_samples} samples")
    """
    path = spec.path
    fmt = spec.format or infer_format(path)

    # Check path exists
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    logger.info(f"Loading data from {path} (format: {fmt.value})")

    # Load raw DataFrame
    df = load_table(path, columns=spec.columns, filters=spec.filters)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns exist
    validate_columns(
        df,
        label_col=spec.label_col,
        id_col=spec.id_col,
        group_col=spec.group_col,
        feature_cols=spec.feature_cols,
    )

    # Extract labels (if specified)
    y = None
    if spec.label_col is not None:
        y = df[spec.label_col].astype(str)

        # Drop rows with missing labels for training
        if y.isna().any():
            n_missing = y.isna().sum()
            logger.warning(f"Dropping {n_missing} rows with missing labels")
            valid_mask = y.notna()
            df = df[valid_mask].copy()
            y = y[valid_mask].copy()

        # Filter to specified classes
        if spec.classes is not None:
            class_mask = y.isin(spec.classes)
            n_filtered = (~class_mask).sum()
            if n_filtered > 0:
                logger.info(f"Filtered to {len(spec.classes)} classes, removed {n_filtered} samples")
                df = df[class_mask].copy()
                y = y[class_mask].copy()

    # Extract IDs (if specified)
    ids = None
    if spec.id_col is not None:
        ids = df[spec.id_col].values

    # Extract groups (if specified)
    groups = None
    if spec.group_col is not None:
        groups = df[spec.group_col].astype(str).values

    # Determine feature columns
    exclude_cols = set()
    if spec.label_col:
        exclude_cols.add(spec.label_col)
    if spec.id_col:
        exclude_cols.add(spec.id_col)
    if spec.group_col:
        exclude_cols.add(spec.group_col)

    if spec.feature_cols is not None:
        feature_cols = spec.feature_cols
    else:
        # Auto-detect numeric columns
        numeric_df = df.drop(columns=list(exclude_cols), errors="ignore").select_dtypes(
            include=[np.number]
        )
        feature_cols = sorted(numeric_df.columns.tolist())

    if len(feature_cols) == 0:
        raise ValueError(
            "No numeric feature columns found. "
            "Specify feature_cols explicitly or ensure data contains numeric columns."
        )

    # Extract feature matrix
    X_df = df[feature_cols].copy()

    # Validate features (warnings only)
    warnings = validate_features(X_df, strict=False)
    for w in warnings:
        logger.warning(w)

    # Convert to numpy array (float32 for memory efficiency)
    X = X_df.values.astype(np.float32)

    # Build metadata DataFrame
    meta_cols = []
    meta_data = {}
    if spec.id_col:
        meta_cols.append(spec.id_col)
        meta_data[spec.id_col] = df[spec.id_col].values
    if spec.label_col:
        meta_cols.append(spec.label_col)
        meta_data[spec.label_col] = df[spec.label_col].values
    if spec.group_col:
        meta_cols.append(spec.group_col)
        meta_data[spec.group_col] = df[spec.group_col].values

    df_meta = pd.DataFrame(meta_data) if meta_data else None

    logger.info(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")
    if y is not None:
        logger.info(f"Classes: {np.unique(y).tolist()}")

    return LoadedDataset(
        X=X,
        y=y.values if y is not None else None,
        feature_names=feature_cols,
        ids=ids,
        groups=groups,
        df_meta=df_meta,
    )


def _load_csv(
    path: Path,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load CSV file."""
    kwargs = {}
    if columns is not None:
        kwargs["usecols"] = columns

    return pd.read_csv(path, **kwargs)


def _load_parquet(
    path: Path,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load single Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def _load_parquet_dataset(
    path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[dict] = None,
    glob_pattern: str = "**/*.parquet",
) -> pd.DataFrame:
    """
    Load Parquet dataset directory (chunked files, optional hive partitioning).

    Parameters
    ----------
    path : Path
        Directory containing parquet files
    columns : List[str], optional
        Columns to load
    filters : dict, optional
        Partition filters (pyarrow filter expressions)
    glob_pattern : str
        Glob pattern for finding parquet files

    Returns
    -------
    pd.DataFrame
        Combined dataframe from all parquet files

    Raises
    ------
    ValueError
        If no parquet files found or schema mismatch
    """
    import pyarrow.dataset as ds

    path = Path(path)

    # Find all parquet files
    parquet_files = list(path.glob(glob_pattern))

    # Filter out hidden/system files
    parquet_files = [
        f for f in parquet_files if not f.name.startswith(".") and not f.name.startswith("_")
    ]

    if not parquet_files:
        raise ValueError(
            f"No parquet files found in {path} matching pattern '{glob_pattern}'. "
            "Ensure the directory contains .parquet files."
        )

    logger.info(f"Found {len(parquet_files)} parquet files in dataset directory")

    # Try to load as a dataset (handles partitioning automatically)
    try:
        dataset = ds.dataset(path, format="parquet")
    except Exception as e:
        # If dataset loading fails, try manual concatenation
        logger.warning(f"PyArrow dataset loading failed: {e}. Trying manual load.")
        return _load_parquet_files_manual(parquet_files, columns)

    # Apply filters if provided
    scanner_kwargs = {}
    if columns is not None:
        scanner_kwargs["columns"] = columns
    if filters is not None:
        scanner_kwargs["filter"] = _dict_to_pyarrow_filter(filters)

    try:
        table = dataset.to_table(**scanner_kwargs)
        return table.to_pandas()
    except Exception as e:
        raise ValueError(
            f"Failed to load parquet dataset: {e}. "
            "This may indicate schema mismatch between parquet files. "
            "Ensure all parquet files have the same schema."
        )


def _load_parquet_files_manual(
    files: List[Path],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Manually load and concatenate parquet files."""
    import pyarrow.parquet as pq

    dfs = []
    reference_schema = None

    for i, f in enumerate(files):
        table = pq.read_table(f, columns=columns)

        # Check schema consistency
        if reference_schema is None:
            reference_schema = table.schema
        else:
            if table.schema != reference_schema:
                raise ValueError(
                    f"Schema mismatch in parquet files. "
                    f"File {f} has different schema than first file. "
                    f"Expected columns: {reference_schema.names}, "
                    f"Got: {table.schema.names}"
                )

        dfs.append(table.to_pandas())

    return pd.concat(dfs, ignore_index=True)


def _dict_to_pyarrow_filter(filters: dict):
    """Convert dict filters to PyArrow filter expression."""
    import pyarrow.compute as pc

    expressions = []
    for col, value in filters.items():
        if isinstance(value, (list, tuple)):
            expressions.append(pc.field(col).isin(value))
        else:
            expressions.append(pc.field(col) == value)

    if len(expressions) == 1:
        return expressions[0]
    elif len(expressions) > 1:
        result = expressions[0]
        for expr in expressions[1:]:
            result = result & expr
        return result
    return None


# Convenience functions for backward compatibility


def load_data_legacy(
    csv_path: Path,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    drop_na_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Legacy interface for loading data (backward compatible with io.loaders.load_data).

    This function maintains the same signature as the original load_data function
    for backward compatibility.

    Parameters
    ----------
    csv_path : Path
        Path to data file (CSV or Parquet)
    label_col : str
        Name of label column
    feature_cols : List[str], optional
        Feature columns; auto-detects numeric if None
    drop_na_labels : bool
        Whether to drop rows with missing labels

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    """
    # Infer format and create spec
    spec = DataSpec(
        path=csv_path,
        label_col=label_col,
        feature_cols=feature_cols,
    )

    # Load dataset
    dataset = load_data(spec)

    # Convert back to legacy format
    X = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    y = pd.Series(dataset.y, name=label_col)

    return X, y
