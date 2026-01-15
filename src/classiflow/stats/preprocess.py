"""Data preprocessing for statistical analysis."""

from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


def select_numeric_features(
    df: pd.DataFrame, label_col: str, whitelist: Optional[List[str]] = None, blacklist: Optional[List[str]] = None
) -> List[str]:
    """Select numeric features from dataframe.

    Args:
        df: Input dataframe
        label_col: Label column name (will be excluded)
        whitelist: If provided, only these features will be included (after numeric filter)
        blacklist: Features to exclude

    Returns:
        List of feature column names
    """
    # Start with all numeric columns except label
    all_cols = df.columns.tolist()
    if label_col in all_cols:
        all_cols.remove(label_col)

    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Apply whitelist
    if whitelist is not None:
        numeric_cols = [c for c in numeric_cols if c in whitelist]

    # Apply blacklist
    if blacklist is not None:
        numeric_cols = [c for c in numeric_cols if c not in blacklist]

    return numeric_cols


def prepare_data(
    df: pd.DataFrame,
    label_col: str,
    classes: Optional[List[str]] = None,
    feature_whitelist: Optional[List[str]] = None,
    feature_blacklist: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Prepare data for statistical analysis.

    Args:
        df: Input dataframe
        label_col: Name of label column
        classes: Optional subset/order of classes to include
        feature_whitelist: Optional list of features to include
        feature_blacklist: Optional list of features to exclude

    Returns:
        Tuple of (processed_df, feature_list, class_list)

    Raises:
        ValueError: If label_col not in dataframe or no valid classes/features found
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    # Copy and drop rows with missing labels
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(str)

    if len(df) == 0:
        raise ValueError("No valid rows after dropping missing labels")

    # Restrict to specified classes if provided
    if classes is not None:
        df = df[df[label_col].isin(classes)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for specified classes: {classes}")
        # Make categorical with specified order
        df[label_col] = pd.Categorical(df[label_col], categories=classes, ordered=True)
        class_list = classes
    else:
        # Use natural order of appearance
        class_list = list(pd.unique(df[label_col]))

    if len(class_list) < 2:
        raise ValueError(f"At least 2 classes required, found {len(class_list)}")

    # Select features
    features = select_numeric_features(
        df, label_col, whitelist=feature_whitelist, blacklist=feature_blacklist
    )

    if len(features) == 0:
        raise ValueError("No numeric features found after filtering")

    return df, features, class_list


def compute_class_stats(
    df: pd.DataFrame, feature: str, label_col: str, classes: List[str]
) -> pd.DataFrame:
    """Compute descriptive statistics per class for a feature.

    Args:
        df: Input dataframe
        feature: Feature column name
        label_col: Label column name
        classes: List of class labels

    Returns:
        DataFrame with columns: class, n, n_missing, mean, sd, median, q25, q75, iqr
    """
    rows = []
    for c in classes:
        x = df.loc[df[label_col] == c, feature]
        x_valid = x.dropna()
        n_valid = len(x_valid)
        n_missing = len(x) - n_valid

        if n_valid > 0:
            mean_val = float(np.mean(x_valid))
            sd_val = float(np.std(x_valid, ddof=1)) if n_valid > 1 else np.nan
            median_val = float(np.median(x_valid))
            q25 = float(np.percentile(x_valid, 25))
            q75 = float(np.percentile(x_valid, 75))
            iqr = q75 - q25
        else:
            mean_val = sd_val = median_val = q25 = q75 = iqr = np.nan

        rows.append(
            {
                "class": c,
                "n": n_valid,
                "n_missing": n_missing,
                "mean": mean_val,
                "sd": sd_val,
                "median": median_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
            }
        )

    return pd.DataFrame(rows)
