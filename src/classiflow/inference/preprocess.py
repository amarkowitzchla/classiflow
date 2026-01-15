"""Feature preprocessing and alignment for inference."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureAligner:
    """
    Align inference data features with training feature schema.

    Handles:
    - Missing features (strict or lenient mode)
    - Extra features (safely dropped)
    - Type coercion
    - Missing value imputation
    """

    def __init__(
        self,
        required_features: List[str],
        strict: bool = True,
        fill_strategy: str = "zero",
        training_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize feature aligner.

        Parameters
        ----------
        required_features : List[str]
            List of features expected by the model
        strict : bool
            If True, fail on missing features; if False, fill them
        fill_strategy : str
            Strategy for filling missing features: "zero" or "median"
        training_stats : Optional[Dict]
            Training statistics (means, medians) for imputation
        """
        self.required_features = required_features
        self.strict = strict
        self.fill_strategy = fill_strategy
        self.training_stats = training_stats or {}

        self.warnings: List[str] = []
        self.errors: List[str] = []

    def align(
        self, df: pd.DataFrame, id_col: Optional[str] = None, label_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Align input dataframe to required features.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        id_col : Optional[str]
            ID column to preserve (won't be used as feature)
        label_col : Optional[str]
            Label column to preserve (won't be used as feature)

        Returns
        -------
        X_aligned : pd.DataFrame
            Feature matrix with exactly required_features columns
        metadata : pd.DataFrame
            Preserved ID and label columns
        warnings : List[str]
            List of warnings encountered
        """
        self.warnings = []
        self.errors = []

        # Preserve metadata columns
        metadata_cols = [c for c in [id_col, label_col] if c is not None and c in df.columns]
        metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

        # Check for missing features
        available_features = [f for f in self.required_features if f in df.columns]
        missing_features = set(self.required_features) - set(df.columns)

        if missing_features:
            msg = f"Missing {len(missing_features)} required features: {sorted(list(missing_features))[:10]}"
            if len(missing_features) > 10:
                msg += f" ... and {len(missing_features) - 10} more"

            if self.strict:
                self.errors.append(msg)
                raise ValueError(
                    f"Strict mode: {msg}. Set strict=False to fill missing features."
                )
            else:
                self.warnings.append(msg)
                logger.warning(msg)

        # Start with available features
        X = df[available_features].copy()

        # Fill missing features
        if not self.strict and missing_features:
            logger.info(f"Filling {len(missing_features)} missing features using strategy: {self.fill_strategy}")
            for feat in missing_features:
                if self.fill_strategy == "zero":
                    X[feat] = 0.0
                elif self.fill_strategy == "median":
                    fill_val = self.training_stats.get(feat, {}).get("median", 0.0)
                    X[feat] = fill_val
                else:
                    X[feat] = 0.0

        # Reorder to match required_features
        X = X[self.required_features]

        # Type coercion: ensure numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                    n_coerced = X[col].isna().sum()
                    if n_coerced > 0:
                        self.warnings.append(
                            f"Coerced {n_coerced} non-numeric values to NaN in column '{col}'"
                        )
                except Exception as e:
                    self.warnings.append(f"Failed to coerce column '{col}' to numeric: {e}")

        # Check for NaN values
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            msg = f"NaN values detected in {len(nan_cols)} columns: {nan_cols[:5]}"
            if len(nan_cols) > 5:
                msg += f" ... and {len(nan_cols) - 5} more"
            self.warnings.append(msg)
            logger.warning(msg)

            # Fill NaN values
            X = X.fillna(0.0)
            logger.info("Filled NaN values with 0.0")

        # Extra features warning
        extra_features = set(df.columns) - set(self.required_features) - set(metadata_cols)
        if extra_features and len(extra_features) < 100:
            logger.debug(f"Dropping {len(extra_features)} extra features not used in training")

        return X, metadata, self.warnings


def validate_input_data(df: pd.DataFrame, id_col: Optional[str] = None) -> List[str]:
    """
    Validate input dataframe for common issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    id_col : Optional[str]
        ID column name

    Returns
    -------
    warnings : List[str]
        List of validation warnings
    """
    warnings = []

    # Check for empty dataframe
    if df.empty:
        warnings.append("Input dataframe is empty")
        return warnings

    # Check for duplicate IDs
    if id_col is not None and id_col in df.columns:
        dup_ids = df[id_col].duplicated()
        if dup_ids.any():
            n_dups = dup_ids.sum()
            warnings.append(f"Found {n_dups} duplicate IDs in '{id_col}' column")

    # Check for all-null columns
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        warnings.append(f"Found {len(null_cols)} columns with all null values: {null_cols[:5]}")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    if inf_cols:
        warnings.append(f"Found infinite values in {len(inf_cols)} columns: {inf_cols[:5]}")

    # Check for extreme values (could indicate data issues)
    for col in numeric_cols:
        if df[col].std() > 1e6:
            warnings.append(f"Column '{col}' has very large standard deviation ({df[col].std():.2e})")

    return warnings


def compute_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute feature statistics for imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Training data

    Returns
    -------
    stats : Dict[str, Dict[str, float]]
        Dictionary mapping feature names to their statistics
    """
    stats = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
            }

    return stats
