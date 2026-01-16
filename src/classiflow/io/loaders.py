"""Data loading and validation functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    data_path: Union[Path, str],
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    drop_na_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix and labels from CSV, Parquet, or Parquet dataset directory.

    Parameters
    ----------
    data_path : Path or str
        Path to data file (.csv, .parquet) or directory (parquet dataset)
    label_col : str
        Name of the label column
    feature_cols : Optional[List[str]]
        Explicit list of feature columns; if None, auto-select numeric columns
    drop_na_labels : bool
        Whether to drop rows with missing labels

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric columns only)
    y : pd.Series
        Label series
    """
    from classiflow.data import load_table

    data_path = Path(data_path)
    logger.info(f"Loading data from {data_path}")
    df = load_table(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available: {list(df.columns)}")

    y = df[label_col].astype(str)

    if drop_na_labels:
        valid = y.notna()
        df = df[valid].copy()
        y = y[valid].copy()

    if feature_cols is not None:
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        X = df[feature_cols].copy()
    else:
        # Auto-select numeric columns excluding label
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    logger.info(f"Loaded data: X shape={X.shape}, y nunique={y.nunique()}")
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validate feature matrix and labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels

    Raises
    ------
    ValueError
        If data is invalid
    """
    if X.shape[0] != len(y):
        raise ValueError(f"X and y length mismatch: X has {X.shape[0]} rows, y has {len(y)}")

    if X.shape[0] < 10:
        raise ValueError(f"Too few samples: {X.shape[0]}. Need at least 10.")

    if X.shape[1] < 1:
        raise ValueError("No features in X.")

    if y.isna().any():
        raise ValueError("y contains NaN values.")

    # Check for constant features (will be removed by VarianceThreshold)
    n_constant = (X.std() == 0).sum()
    if n_constant > 0:
        logger.warning(f"{n_constant} constant features detected (will be removed by VarianceThreshold)")

    # Check for missing values
    na_cols = X.columns[X.isna().any()].tolist()
    if na_cols:
        logger.warning(f"{len(na_cols)} feature columns have missing values: {na_cols[:5]}")

    logger.info("Data validation passed.")
