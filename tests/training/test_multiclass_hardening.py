"""Regression tests for multiclass technical validation hardening."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from classiflow.models import get_estimators
from classiflow.splitting import assert_no_patient_leakage, iter_outer_splits
from classiflow.training.multiclass import _compute_multiclass_metrics


def _make_grouped_multiclass_dataset(
    n_classes: int = 8,
    patients_per_class: int = 25,
    samples_per_patient: int = 10,
    n_features: int = 50,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    rng = np.random.RandomState(seed)
    n_patients = n_classes * patients_per_class
    class_means = rng.normal(0, 2.0, size=(n_classes, n_features))

    rows = []
    labels = []
    groups = []
    for cls in range(n_classes):
        for p_idx in range(patients_per_class):
            patient_id = f"p{cls:02d}_{p_idx:03d}"
            for _ in range(samples_per_patient):
                rows.append(class_means[cls] + rng.normal(0, 1.0, size=n_features))
                labels.append(cls)
                groups.append(patient_id)

    X = pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(labels, name="label")
    group_series = pd.Series(groups, name="patient_id")
    class_names = [f"class_{i}" for i in range(n_classes)]
    return X, y, group_series, class_names


def test_multiclass_group_stratified_split_coverage():
    X, y, groups, _ = _make_grouped_multiclass_dataset()
    df_groups = pd.DataFrame({"patient_id": groups}, index=X.index)
    label_ids = list(range(y.nunique()))

    splits = list(
        iter_outer_splits(
            df=df_groups,
            y=y,
            patient_col="patient_id",
            n_splits=3,
            random_state=17,
            stratify=True,
        )
    )
    assert splits, "Expected stratified group splits"

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, 1):
        assert_no_patient_leakage(df_groups, "patient_id", tr_idx, va_idx, f"fold {fold_idx}")
        y_va = y.iloc[va_idx]
        val_counts = y_va.value_counts().reindex(label_ids, fill_value=0)
        assert (val_counts > 0).all(), f"Missing class in fold {fold_idx}: {val_counts.to_dict()}"


def test_multiclass_metrics_no_label_warnings():
    class DummyEstimator:
        def __init__(self, classes, y_pred, proba):
            self.classes_ = classes
            self._y_pred = y_pred
            self._proba = proba

        def predict(self, X):
            return self._y_pred

        def predict_proba(self, X):
            return self._proba

    X = pd.DataFrame(np.zeros((6, 2)))
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 2, 1, 2, 0, 1])
    proba = np.array(
        [
            [0.7, 0.3],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.3, 0.7],
        ]
    )
    estimator = DummyEstimator(classes=[0, 1], y_pred=y_pred, proba=proba)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _, _, aligned_proba = _compute_multiclass_metrics(estimator, X, y_true, [0, 1, 2], return_preds=True)

    assert aligned_proba.shape == (len(y_true), 3)


def test_multiclass_metrics_include_decision_keys():
    class DummyEstimator:
        def __init__(self, y_pred, proba):
            self.classes_ = [0, 1]
            self._y_pred = y_pred
            self._proba = proba

        def predict(self, X):
            return self._y_pred

        def predict_proba(self, X):
            return self._proba

    X = pd.DataFrame(np.zeros((8, 2)))
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.3, 0.7],
        ]
    )
    estimator = DummyEstimator(y_pred=y_pred, proba=proba)

    metrics = _compute_multiclass_metrics(estimator, X, y_true, [0, 1])
    for key in ["sensitivity", "specificity", "ppv", "npv", "recall", "precision", "mcc"]:
        assert key in metrics, f"missing expected metric: {key}"


def test_multiclass_logreg_convergence_warning_free():
    X, y, _, _ = _make_grouped_multiclass_dataset()
    logreg_params = {
        "solver": "saga",
        "penalty": "l2",
        "max_iter": 5000,
        "tol": 1e-3,
        "C": 1.0,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    estimators = get_estimators(random_state=7, max_iter=1000, logreg_params=logreg_params)
    clf = estimators["LogisticRegression"]

    assert isinstance(clf, LogisticRegression)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipe = StandardScaler()
        X_scaled = pipe.fit_transform(X)
        clf.fit(X_scaled, y)
