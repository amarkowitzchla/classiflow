"""Group-aware stratified splitting utilities for patient-safe CV."""

from __future__ import annotations

import logging
from typing import Iterator, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None


def make_group_labels(df: pd.DataFrame, patient_col: str, label_col: str) -> pd.Series:
    """
    Return a single label per patient ID.

    Raises
    ------
    ValueError
        If any patient has multiple labels.
    """
    if patient_col not in df.columns:
        raise ValueError(f"Patient column '{patient_col}' not found in data.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data.")

    subset = df[[patient_col, label_col]].dropna()

    label_counts = subset.groupby(patient_col)[label_col].nunique(dropna=False)
    conflicting = label_counts[label_counts > 1].index.tolist()
    if conflicting:
        raise ValueError(
            "Patient label conflict for the following IDs: "
            f"{sorted(map(str, conflicting))}"
        )

    return subset.groupby(patient_col)[label_col].first()


def iter_outer_splits(
    df: pd.DataFrame,
    y: Iterable,
    patient_col: str,
    n_splits: int,
    random_state: int,
    mode: str = "kfold",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield outer split indices using patient-level stratification.
    """
    if mode != "kfold":
        raise ValueError(f"Unsupported split mode: {mode}")

    y_series = _as_series(y, df.index, name="label")
    patient_df = pd.DataFrame({patient_col: df[patient_col].values, "label": y_series.values})
    patient_labels = make_group_labels(patient_df, patient_col, "label")
    patient_ids = patient_labels.index.to_numpy()
    y_patient = patient_labels.values

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        X_dummy = np.zeros((len(patient_ids), 1), dtype=np.float32)
        for tr_pat_idx, va_pat_idx in splitter.split(X_dummy, y_patient, groups=patient_ids):
            tr_patients = patient_ids[tr_pat_idx]
            va_patients = patient_ids[va_pat_idx]
            yield _expand_patient_indices(df, patient_col, tr_patients, va_patients)
        return

    folds = _greedy_patient_folds(patient_ids, y_patient, n_splits, random_state)
    for fold_idx in range(n_splits):
        va_patients = np.array(folds[fold_idx], dtype=object)
        tr_patients = np.array([pid for i, f in enumerate(folds) if i != fold_idx for pid in f], dtype=object)
        yield _expand_patient_indices(df, patient_col, tr_patients, va_patients)


def iter_inner_splits(
    df_tr: pd.DataFrame,
    y_tr: Iterable,
    patient_col: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield inner split indices using patient-level stratification.
    """
    for repeat in range(n_repeats):
        seed = random_state + repeat
        for tr_idx, va_idx in iter_outer_splits(
            df=df_tr,
            y=y_tr,
            patient_col=patient_col,
            n_splits=n_splits,
            random_state=seed,
            mode="kfold",
        ):
            yield tr_idx, va_idx


def assert_no_patient_leakage(
    df: pd.DataFrame,
    patient_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    context: str,
) -> None:
    """Raise if any patient appears in both train and validation sets."""
    train_patients = set(df.iloc[train_idx][patient_col].astype(str).dropna())
    val_patients = set(df.iloc[val_idx][patient_col].astype(str).dropna())
    overlap = sorted(train_patients.intersection(val_patients))
    if overlap:
        raise ValueError(
            f"Patient leakage detected ({context}): "
            f"{overlap}"
        )


def _as_series(values: Iterable, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values, index=index, name=name)


def _expand_patient_indices(
    df: pd.DataFrame,
    patient_col: str,
    train_patients: np.ndarray,
    val_patients: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = df[patient_col].isin(train_patients).values
    val_mask = df[patient_col].isin(val_patients).values
    tr_idx = np.flatnonzero(train_mask)
    va_idx = np.flatnonzero(val_mask)
    return tr_idx, va_idx


def _greedy_patient_folds(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[List[object]]:
    rng = np.random.RandomState(random_state)
    folds: List[List[object]] = [[] for _ in range(n_splits)]
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_ids = patient_ids[labels == label]
        label_ids = label_ids.copy()
        rng.shuffle(label_ids)
        for idx, pid in enumerate(label_ids):
            folds[idx % n_splits].append(pid)
    return folds
