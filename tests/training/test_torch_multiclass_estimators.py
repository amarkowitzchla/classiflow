"""Tests for sklearn-compatible torch multiclass estimators."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classiflow.models.torch_multiclass import TorchLinearClassifier, TorchMLPClassifier


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(int)


def test_torch_linear_fit_predict_proba() -> None:
    X, y = _make_data(1)
    clf = TorchLinearClassifier(
        epochs=3,
        batch_size=16,
        device="cpu",
        random_state=7,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_torch_mlp_gridsearch_pipeline() -> None:
    X, y = _make_data(2)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", TorchMLPClassifier(
            epochs=3,
            batch_size=16,
            hidden_dim=32,
            device="cpu",
            random_state=11,
        )),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__lr": [1e-2, 1e-3]},
        cv=2,
        n_jobs=1,
    )
    grid.fit(X, y)
    proba = grid.best_estimator_.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
