"""Dataset container types for Classiflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class LoadedDataset:
    """
    Container for a loaded and prepared dataset.

    This class holds the feature matrix, labels, and associated metadata
    after loading and preprocessing a dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features), dtype float32
    y : np.ndarray, optional
        Label array of shape (n_samples,); None for inference-only datasets
    feature_names : List[str]
        Names of features in X (column order)
    ids : np.ndarray, optional
        Sample IDs if id_col was specified
    groups : np.ndarray, optional
        Group IDs for stratification (e.g., patient IDs)
    df_meta : pd.DataFrame, optional
        DataFrame with label/id/group columns retained for exports

    Attributes
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of unique classes (0 if y is None)
    class_names : List[str]
        Sorted list of unique class names

    Examples
    --------
    >>> dataset = load_data(spec)
    >>> print(f"Loaded {dataset.n_samples} samples with {dataset.n_features} features")
    >>> X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y)
    """

    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    ids: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    df_meta: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Validate dataset consistency."""
        # Ensure X is float32
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32)

        # Validate dimensions
        if self.y is not None and len(self.y) != self.X.shape[0]:
            raise ValueError(
                f"X and y length mismatch: X has {self.X.shape[0]} rows, y has {len(self.y)}"
            )

        if self.ids is not None and len(self.ids) != self.X.shape[0]:
            raise ValueError(
                f"X and ids length mismatch: X has {self.X.shape[0]} rows, ids has {len(self.ids)}"
            )

        if self.groups is not None and len(self.groups) != self.X.shape[0]:
            raise ValueError(
                f"X and groups length mismatch: X has {self.X.shape[0]} rows, groups has {len(self.groups)}"
            )

        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) does not match "
                f"X columns ({self.X.shape[1]})"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        """Number of unique classes (0 if no labels)."""
        if self.y is None:
            return 0
        return len(np.unique(self.y))

    @property
    def class_names(self) -> List[str]:
        """Sorted list of unique class names."""
        if self.y is None:
            return []
        return sorted(list(np.unique(self.y)))

    @property
    def class_counts(self) -> dict:
        """Dictionary of class counts."""
        if self.y is None:
            return {}
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))

    def to_dataframe(self, include_meta: bool = True) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame.

        Parameters
        ----------
        include_meta : bool
            Whether to include id/label/group columns

        Returns
        -------
        pd.DataFrame
            DataFrame with features and optionally metadata
        """
        df = pd.DataFrame(self.X, columns=self.feature_names)

        if include_meta:
            if self.ids is not None:
                df.insert(0, "_id", self.ids)
            if self.y is not None:
                df["_label"] = self.y
            if self.groups is not None:
                df["_group"] = self.groups

        return df

    def get_X_df(self) -> pd.DataFrame:
        """Get feature matrix as DataFrame with feature names."""
        return pd.DataFrame(self.X, columns=self.feature_names)

    def get_y_series(self, name: str = "label") -> Optional[pd.Series]:
        """Get labels as Series."""
        if self.y is None:
            return None
        return pd.Series(self.y, name=name)

    def subset(self, indices: np.ndarray) -> LoadedDataset:
        """
        Create a subset of the dataset.

        Parameters
        ----------
        indices : np.ndarray
            Indices to select

        Returns
        -------
        LoadedDataset
            New dataset with selected samples
        """
        return LoadedDataset(
            X=self.X[indices],
            y=self.y[indices] if self.y is not None else None,
            feature_names=self.feature_names.copy(),
            ids=self.ids[indices] if self.ids is not None else None,
            groups=self.groups[indices] if self.groups is not None else None,
            df_meta=self.df_meta.iloc[indices].copy() if self.df_meta is not None else None,
        )
