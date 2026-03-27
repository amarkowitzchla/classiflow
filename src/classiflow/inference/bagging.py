"""Helpers for inspecting bagged estimators during inference."""

from __future__ import annotations

from copy import copy
from typing import Any, Iterator, Optional

from sklearn.ensemble import BaggingClassifier


def get_bagging_estimator(model: Any) -> Optional[tuple[Optional[str], BaggingClassifier]]:
    """Return the fitted BaggingClassifier for a model, if present."""
    if isinstance(model, BaggingClassifier):
        return None, model

    steps = getattr(model, "steps", None)
    if steps:
        step_name, estimator = steps[-1]
        if isinstance(estimator, BaggingClassifier):
            return str(step_name), estimator

    return None


def get_bagging_member_count(model: Any) -> int:
    """Return the number of fitted bag members for a model."""
    located = get_bagging_estimator(model)
    if not located:
        return 0

    _, bagger = located
    estimators = getattr(bagger, "estimators_", None)
    return len(estimators or [])


def iter_single_member_models(model: Any) -> Iterator[tuple[int, Any, str]]:
    """Yield model clones that score with a single bag member at a time."""
    located = get_bagging_estimator(model)
    if not located:
        return

    step_name, bagger = located
    estimators = getattr(bagger, "estimators_", None) or []
    for idx, estimator in enumerate(estimators, start=1):
        single_bagger = _build_single_member_bagger(bagger, idx - 1)
        if step_name is None:
            yield idx, single_bagger, _qualname(estimator)
            continue

        cloned_model = copy(model)
        cloned_steps = list(getattr(model, "steps", []))
        cloned_steps[-1] = (step_name, single_bagger)
        cloned_model.steps = cloned_steps
        yield idx, cloned_model, _qualname(estimator)


def _build_single_member_bagger(bagger: BaggingClassifier, index: int) -> BaggingClassifier:
    """Clone a fitted bagger so it predicts with one member only."""
    single = copy(bagger)
    single.n_estimators = 1
    single.estimators_ = [bagger.estimators_[index]]

    if hasattr(bagger, "estimators_features_"):
        single.estimators_features_ = [bagger.estimators_features_[index]]

    if hasattr(bagger, "estimators_samples_"):
        try:
            single.estimators_samples_ = [bagger.estimators_samples_[index]]
        except Exception:
            pass

    return single


def _qualname(obj: Any) -> str:
    """Return a stable fully qualified class name for an object."""
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"
