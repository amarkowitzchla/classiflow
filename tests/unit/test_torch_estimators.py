"""Unit tests for torch estimators."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from classiflow.backends.torch.estimators import (
    TorchLogisticRegressionClassifier,
    TorchSoftmaxRegressionClassifier,
)


def _make_binary_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(32, 6)).astype(np.float32)
    y = (rng.uniform(size=32) > 0.5).astype(int)
    return X, y


def _make_multiclass_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    return X, y


def test_torch_binary_predict_proba_shape() -> None:
    X, y = _make_binary_data()
    clf = TorchLogisticRegressionClassifier(
        epochs=5,
        patience=0,
        val_fraction=0.0,
        device="cpu",
        num_workers=0,
        seed=1,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_torch_binary_deterministic_seed() -> None:
    X, y = _make_binary_data()
    clf1 = TorchLogisticRegressionClassifier(
        epochs=5,
        patience=0,
        val_fraction=0.0,
        device="cpu",
        num_workers=0,
        seed=7,
    )
    clf2 = TorchLogisticRegressionClassifier(
        epochs=5,
        patience=0,
        val_fraction=0.0,
        device="cpu",
        num_workers=0,
        seed=7,
    )
    clf1.fit(X, y)
    clf2.fit(X, y)
    proba1 = clf1.predict_proba(X)
    proba2 = clf2.predict_proba(X)
    assert np.allclose(proba1, proba2, atol=1e-4)


def test_torch_multiclass_predict_proba_shape() -> None:
    X, y = _make_multiclass_data()
    clf = TorchSoftmaxRegressionClassifier(
        epochs=5,
        patience=0,
        val_fraction=0.0,
        device="cpu",
        num_workers=0,
        seed=3,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
