"""Tests validating MetaPredictor outputs."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from classiflow.inference.predict import MetaPredictor


def _make_meta_dataset():
    X = pd.DataFrame({"meta_score": [0.0, 0.2, 0.8, 1.0]})
    y = [0, 0, 1, 1]
    return X, y


def test_meta_predictor_outputs_probabilities():
    X, y = _make_meta_dataset()
    model = LogisticRegression(solver="liblinear", random_state=0)
    _ = model.fit(X, y)

    binary_predictions = pd.DataFrame({"meta_score": X["meta_score"].values})
    calibration_metadata = {
        "method_used": "sigmoid",
        "enabled": True,
        "cv": 2,
        "bins": 10,
        "warnings": [],
    }

    predictor = MetaPredictor(
        meta_model=model,
        meta_features=["meta_score"],
        meta_classes=["0", "1"],
        calibration_metadata=calibration_metadata,
    )

    predictions = predictor.predict(binary_predictions)

    assert "y_prob" in predictions.columns
    assert predictions["y_prob"].between(0.0, 1.0).all()
    assert "y_score_raw" in predictions.columns
    assert predictions["y_score_raw"].between(0.0, 1.0).all()
    assert predictions["calibration_method"].iloc[0] == "sigmoid"
    assert predictions["calibration_enabled"].iloc[0]
    assert "y_prob_0" in predictions.columns and "y_prob_1" in predictions.columns
