"""Data specification types for Classiflow data loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class DataFormat(str, Enum):
    """Supported data formats."""

    CSV = "csv"
    PARQUET = "parquet"
    PARQUET_DATASET = "parquet_dataset"

    @classmethod
    def from_path(cls, path: Path) -> DataFormat:
        """
        Infer format from path.

        Parameters
        ----------
        path : Path
            Path to data file or directory

        Returns
        -------
        DataFormat
            Inferred format

        Raises
        ------
        ValueError
            If format cannot be inferred
        """
        path = Path(path)

        if path.is_dir():
            return cls.PARQUET_DATASET

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls.CSV
        elif suffix == ".parquet":
            return cls.PARQUET
        else:
            raise ValueError(
                f"Cannot infer data format from path: {path}. "
                f"Expected .csv, .parquet file, or directory for parquet dataset."
            )


@dataclass
class DataSpec:
    """
    Specification for loading a dataset.

    This dataclass contains all the information needed to load and prepare
    a dataset for training or inference.

    Parameters
    ----------
    path : Path
        Path to data file (.csv, .parquet) or directory (parquet dataset)
    format : DataFormat, optional
        Data format; inferred from path if not specified
    label_col : str, optional
        Name of label column (required for training, optional for inference)
    id_col : str, optional
        Name of sample ID column (preserved in metadata)
    group_col : str, optional
        Name of grouping column (e.g., patient_id for leakage protection)
    feature_cols : List[str], optional
        Explicit list of feature columns; if None, infers numeric columns
    classes : List[str], optional
        Subset/order of classes to include
    filters : Dict[str, Any], optional
        Filters for dataset directory (future-proof; for partition filters)
    dataset_glob : str, optional
        Glob pattern for parquet dataset (default: **/*.parquet)
    columns : List[str], optional
        Subset of columns to load (performance optimization)

    Examples
    --------
    >>> # Single parquet file
    >>> spec = DataSpec(
    ...     path=Path("data.parquet"),
    ...     label_col="subtype",
    ... )

    >>> # CSV with explicit features
    >>> spec = DataSpec(
    ...     path=Path("data.csv"),
    ...     label_col="diagnosis",
    ...     feature_cols=["gene1", "gene2", "gene3"],
    ... )

    >>> # Parquet dataset directory
    >>> spec = DataSpec(
    ...     path=Path("data_parquet/"),
    ...     label_col="subtype",
    ...     group_col="patient_id",
    ... )
    """

    path: Path
    format: Optional[DataFormat] = None
    label_col: Optional[str] = None
    id_col: Optional[str] = None
    group_col: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    classes: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = field(default_factory=dict)
    dataset_glob: str = "**/*.parquet"
    columns: Optional[List[str]] = None

    def __post_init__(self):
        """Validate and normalize specification."""
        # Ensure path is Path object
        self.path = Path(self.path)

        # Infer format if not specified
        if self.format is None:
            self.format = DataFormat.from_path(self.path)

        # Ensure filters is a dict
        if self.filters is None:
            self.filters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary for serialization."""
        return {
            "path": str(self.path),
            "format": self.format.value if self.format else None,
            "label_col": self.label_col,
            "id_col": self.id_col,
            "group_col": self.group_col,
            "feature_cols": self.feature_cols,
            "classes": self.classes,
            "filters": self.filters,
            "dataset_glob": self.dataset_glob,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DataSpec:
        """Create spec from dictionary."""
        d = d.copy()
        if "path" in d:
            d["path"] = Path(d["path"])
        if "format" in d and d["format"] is not None:
            d["format"] = DataFormat(d["format"])
        return cls(**d)

    def with_label_col(self, label_col: str) -> DataSpec:
        """Return a new spec with updated label column."""
        return DataSpec(
            path=self.path,
            format=self.format,
            label_col=label_col,
            id_col=self.id_col,
            group_col=self.group_col,
            feature_cols=self.feature_cols,
            classes=self.classes,
            filters=self.filters,
            dataset_glob=self.dataset_glob,
            columns=self.columns,
        )
