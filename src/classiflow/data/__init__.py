"""
Unified data loading layer for Classiflow.

This module provides a modular DataSource / Adapter layer that supports:
- CSV files (*.csv)
- Single Parquet files (*.parquet) - recommended for performance
- Parquet dataset directories (chunked parquet files, optional hive partitioning)

Example usage:
    from classiflow.data import DataSpec, load_data, load_table

    # Load via DataSpec
    spec = DataSpec(
        path=Path("data.parquet"),
        label_col="subtype",
    )
    dataset = load_data(spec)
    X, y = dataset.X, dataset.y

    # Simple table load
    df = load_table(Path("data.parquet"))
"""

from classiflow.data.spec import DataSpec, DataFormat
from classiflow.data.dataset import LoadedDataset
from classiflow.data.loaders import (
    load_table,
    load_data,
    infer_format,
    validate_parquet_available,
)
from classiflow.data.validation import (
    validate_columns,
    validate_features,
    generate_missingness_report,
)

__all__ = [
    # Core types
    "DataSpec",
    "DataFormat",
    "LoadedDataset",
    # Loaders
    "load_table",
    "load_data",
    "infer_format",
    "validate_parquet_available",
    # Validation
    "validate_columns",
    "validate_features",
    "generate_missingness_report",
]
