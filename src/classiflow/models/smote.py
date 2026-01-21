"""Adaptive SMOTE that handles small minority classes gracefully."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance classes.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (should be integers 0-indexed)
    k_neighbors : int
        Number of neighbors for SMOTE
    random_state : int
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled X and y, or original if SMOTE not applicable
    """
    y_series = pd.Series(y)
    vc = y_series.value_counts()

    if len(vc) < 2:
        logger.debug("Less than 2 classes, skipping SMOTE")
        return X, y

    minority = int(vc.min())

    if minority <= 1:
        logger.debug(f"Minority class too small ({minority}), skipping SMOTE")
        return X, y

    # Adapt k_neighbors
    k = max(1, min(k_neighbors, minority - 1))

    if k < 1:
        logger.debug("Cannot determine valid k_neighbors, skipping SMOTE")
        return X, y

    try:
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        logger.debug(f"Applied SMOTE with k={k}: {len(y)} -> {len(y_res)} samples")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"SMOTE failed: {e}. Returning original data.")
        return X, y


class AdaptiveSMOTE:
    """
    SMOTE wrapper that adapts k_neighbors to the current training split size.

    This prevents failures when the minority class is too small for standard SMOTE.
    If the minority count is insufficient, it passes through without resampling.

    This class is compatible with imblearn.pipeline.Pipeline and GridSearchCV.

    Parameters
    ----------
    k_max : int
        Maximum k_neighbors to use (will be adapted down if needed)
    random_state : Optional[int]
        Random seed for reproducibility

    Examples
    --------
    >>> from imblearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = Pipeline([
    ...     ("sampler", AdaptiveSMOTE(k_max=5, random_state=42)),
    ...     ("clf", LogisticRegression())
    ... ])
    """

    def __init__(self, k_max: int = 5, random_state: Optional[int] = None):
        self.k_max = int(k_max)
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (for sklearn compatibility)."""
        return {"k_max": self.k_max, "random_state": self.random_state}

    def set_params(self, **params) -> AdaptiveSMOTE:
        """Set parameters (for sklearn compatibility)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit_resample(self, X, y):
        """
        Fit and resample data using adaptive k_neighbors.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Labels (classification targets)

        Returns
        -------
        X_resampled, y_resampled : array-like
            Resampled data or original data if SMOTE not applicable
        """
        y_series = pd.Series(y)
        vc = y_series.value_counts()
        minority = int(vc.min())

        if minority <= 1:
            logger.debug(f"Minority class too small ({minority}), passing through")
            return X, y

        # Adapt k_neighbors
        k = max(1, min(self.k_max, minority - 1))

        if k < 1:
            logger.debug("Cannot determine valid k_neighbors, passing through")
            return X, y

        try:
            sm = SMOTE(k_neighbors=k, random_state=self.random_state)
            X_res, y_res = sm.fit_resample(X, y)
            logger.debug(f"Applied SMOTE with k={k}: {len(y)} -> {len(y_res)} samples")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Passing through original data.")
            return X, y
