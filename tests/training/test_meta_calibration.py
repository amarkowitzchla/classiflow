"""Tests covering meta-classifier calibration logic."""

from types import SimpleNamespace

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from classiflow.training.meta import _calibrate_meta_classifier


def _make_config(method: str, calibrate: bool = True, min_samples: int = 10):
    return SimpleNamespace(
        calibrate_meta=calibrate,
        calibration_method=method,
        calibration_cv=2,
        calibration_bins=5,
        calibration_isotonic_min_samples=min_samples,
    )


def _build_dummy_meta():
    X = pd.DataFrame({"meta_score": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression(solver="liblinear", random_state=0)
    model.fit(X, y)
    return X, y, model


def test_sigmoid_calibration_enabled():
    X, y, model = _build_dummy_meta()
    config = _make_config("sigmoid")
    calibrated, metadata = _calibrate_meta_classifier(
        model, X, y, config, enabled_requested="true"
    )

    assert isinstance(calibrated, CalibratedClassifierCV)
    assert metadata["enabled"]
    assert metadata["method_used"] == "sigmoid"


def test_isotonic_calibration_fallback():
    X, y, model = _build_dummy_meta()
    config = _make_config("isotonic", min_samples=100)
    _, metadata = _calibrate_meta_classifier(model, X, y, config, enabled_requested="true")

    assert metadata["method_used"] == "sigmoid"
    assert any("Isotonic" in warning for warning in metadata["warnings"])


def test_calibration_disabled():
    X, y, model = _build_dummy_meta()
    config = _make_config("sigmoid")
    config.calibrate_meta = False
    _, metadata = _calibrate_meta_classifier(model, X, y, config, enabled_requested="false")

    assert not metadata["enabled"]
    assert any("disabled" in warning.lower() for warning in metadata["warnings"])
