"""Data validation utilities for Classiflow."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_columns(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
    group_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Validate that required columns exist in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    label_col : str, optional
        Label column name
    id_col : str, optional
        ID column name
    group_col : str, optional
        Group column name
    feature_cols : List[str], optional
        Feature column names

    Returns
    -------
    List[str]
        List of validation errors (empty if all valid)

    Raises
    ------
    ValueError
        If any required column is missing
    """
    errors = []
    available = set(df.columns)

    if label_col is not None and label_col not in available:
        errors.append(f"Label column '{label_col}' not found. Available: {sorted(available)[:10]}...")

    if id_col is not None and id_col not in available:
        errors.append(f"ID column '{id_col}' not found. Available: {sorted(available)[:10]}...")

    if group_col is not None and group_col not in available:
        errors.append(f"Group column '{group_col}' not found. Available: {sorted(available)[:10]}...")

    if feature_cols is not None:
        missing = set(feature_cols) - available
        if missing:
            errors.append(f"Feature columns not found: {sorted(missing)[:10]}")

    if errors:
        raise ValueError("\n".join(errors))

    return errors


def validate_features(
    X: pd.DataFrame,
    strict: bool = True,
) -> List[str]:
    """
    Validate feature matrix for common issues.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    strict : bool
        If True, raise on errors; if False, return warnings

    Returns
    -------
    List[str]
        List of warnings/errors found

    Raises
    ------
    ValueError
        If strict=True and critical errors found
    """
    warnings = []
    errors = []

    # Check for non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        msg = f"{len(non_numeric)} non-numeric feature columns: {non_numeric[:5]}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Check for infinite values
    inf_cols = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
    if inf_cols:
        msg = f"{len(inf_cols)} columns contain infinite values: {inf_cols[:5]}"
        errors.append(msg)

    # Check for constant features
    numeric_cols = X.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) > 0:
        std = numeric_cols.std()
        constant_cols = std[std == 0].index.tolist()
        if constant_cols:
            warnings.append(
                f"{len(constant_cols)} constant features (zero variance): {constant_cols[:5]}"
            )

    # Check for missing values
    na_cols = X.columns[X.isna().any()].tolist()
    if na_cols:
        warnings.append(f"{len(na_cols)} features have missing values: {na_cols[:5]}")

    # Check for duplicate columns
    if X.columns.duplicated().any():
        dup_cols = X.columns[X.columns.duplicated()].tolist()
        errors.append(f"Duplicate column names: {dup_cols[:5]}")

    if errors and strict:
        raise ValueError("Feature validation failed:\n" + "\n".join(errors))

    return warnings + errors


def generate_missingness_report(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a report on missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str], optional
        Feature columns to check; if None, checks all numeric columns

    Returns
    -------
    Dict[str, Any]
        Report containing:
        - total_missing: total count of missing values
        - cols_with_missing: list of columns with missing values
        - missing_counts: dict of {column: count}
        - missing_pct: dict of {column: percentage}
        - rows_with_missing: number of rows with any missing value
    """
    if feature_cols is not None:
        df_check = df[feature_cols]
    else:
        df_check = df.select_dtypes(include=[np.number])

    # Calculate missing counts
    missing_counts = df_check.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    # Calculate percentages
    missing_pct = (cols_with_missing / len(df_check) * 100).round(2)

    # Rows with any missing
    rows_with_missing = df_check.isna().any(axis=1).sum()

    report = {
        "total_missing": int(missing_counts.sum()),
        "cols_with_missing": cols_with_missing.index.tolist(),
        "n_cols_with_missing": len(cols_with_missing),
        "missing_counts": cols_with_missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "rows_with_missing": int(rows_with_missing),
        "rows_with_missing_pct": round(rows_with_missing / len(df_check) * 100, 2),
        "total_cells": int(df_check.shape[0] * df_check.shape[1]),
    }

    return report


def validate_labels(
    y: pd.Series,
    min_samples_per_class: int = 2,
    min_classes: int = 2,
) -> List[str]:
    """
    Validate label series for training.

    Parameters
    ----------
    y : pd.Series
        Label series
    min_samples_per_class : int
        Minimum samples required per class
    min_classes : int
        Minimum number of classes required

    Returns
    -------
    List[str]
        List of validation warnings/errors

    Raises
    ------
    ValueError
        If critical errors found
    """
    errors = []
    warnings = []

    # Check for missing labels
    n_missing = y.isna().sum()
    if n_missing > 0:
        warnings.append(f"{n_missing} samples have missing labels")

    y_clean = y.dropna()

    # Check number of classes
    n_classes = y_clean.nunique()
    if n_classes < min_classes:
        errors.append(f"Only {n_classes} classes found (minimum: {min_classes})")

    # Check samples per class
    class_counts = y_clean.value_counts()
    small_classes = class_counts[class_counts < min_samples_per_class]
    if len(small_classes) > 0:
        errors.append(
            f"Classes with < {min_samples_per_class} samples: "
            f"{small_classes.to_dict()}"
        )

    # Check class imbalance
    if len(class_counts) > 0:
        max_count = class_counts.max()
        min_count = class_counts.min()
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 10:
                warnings.append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
                )

    if errors:
        raise ValueError("Label validation failed:\n" + "\n".join(errors))

    return warnings
