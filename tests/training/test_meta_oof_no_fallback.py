import numpy as np
import pandas as pd

from classiflow.config import MetaConfig
import classiflow.training.meta as meta
from classiflow.training.meta import (
    _build_meta_features,
    _cross_val_scores,
    _filter_meta_training_rows,
)


class DummyPipe:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


class FeatureDrivenPipe:
    def predict_proba(self, X):
        xvals = pd.to_numeric(X["x"], errors="coerce").astype(float).to_numpy()
        pos = np.clip(0.1 + 0.2 * xvals, 0.0, 1.0)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


def test_cross_val_scores_no_fallback():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    scores = _cross_val_scores(DummyPipe(), X, y, [])
    assert np.isnan(scores).all()


def test_meta_features_drop_missing_oof_rows(monkeypatch):
    X = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    y = pd.Series(["A", "B", "A", "B"], index=X.index)
    config = MetaConfig(inner_splits=2, inner_repeats=1, random_state=0)

    tasks = {"task": lambda s: pd.Series([1, 0, 1, 0], index=s.index)}
    best_models = {"task": "model"}
    best_pipes = {"task__model": DummyPipe()}

    monkeypatch.setattr(meta, "_inner_cv_splits_for_task", lambda *args, **kwargs: [])

    meta_features = _build_meta_features(
        X,
        y,
        best_pipes,
        best_models,
        tasks,
        config=config,
        groups_tr=None,
        use_oof=True,
        fold=1,
        variant="none",
    )

    assert meta_features["task_score"].isna().all()

    filtered_X, filtered_y, _ = _filter_meta_training_rows(meta_features, y, None, 1, "none")
    assert filtered_X.empty
    assert filtered_y.empty


def test_validation_meta_features_do_not_depend_on_labels():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=[10, 11, 12, 13])
    y_first = pd.Series(["A", "B", "C", "A"], index=X.index)
    y_second = pd.Series(["C", "C", "B", "B"], index=X.index)
    config = MetaConfig(inner_splits=2, inner_repeats=1, random_state=0)

    tasks = {"A_vs_B": lambda s: s.map({"A": 1.0, "B": 0.0})}
    best_models = {"A_vs_B": "model"}
    best_pipes = {"A_vs_B__model": FeatureDrivenPipe()}

    meta_first = _build_meta_features(
        X,
        y_first,
        best_pipes,
        best_models,
        tasks,
        config=config,
        use_oof=False,
    )
    meta_second = _build_meta_features(
        X,
        y_second,
        best_pipes,
        best_models,
        tasks,
        config=config,
        use_oof=False,
    )

    assert np.allclose(meta_first["A_vs_B_score"].values, meta_second["A_vs_B_score"].values)
    assert meta_first["A_vs_B_score"].notna().all()
