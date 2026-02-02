import numpy as np
import pandas as pd

from classiflow.config import MetaConfig
import classiflow.training.meta as meta
from classiflow.training.meta import _build_meta_features, _cross_val_scores, _filter_meta_training_rows


class DummyPipe:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


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
