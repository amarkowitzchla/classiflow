"""Helpers for bagged estimator strategies."""

from __future__ import annotations

from typing import Any, Dict, Literal

from sklearn.ensemble import BaggingClassifier

EstimatorStrategy = Literal["single", "bagged"]


def is_bagging_enabled(strategy: str) -> bool:
    """Return True when bagging should be applied."""
    return str(strategy).strip().lower() == "bagged"


def wrap_estimator_for_strategy(
    estimator: Any,
    *,
    strategy: str,
    random_state: int,
    bagging_n_estimators: int,
    bagging_max_samples: float,
    bagging_max_features: float,
    bagging_bootstrap: bool,
    bagging_bootstrap_features: bool,
) -> Any:
    """Wrap an estimator in a BaggingClassifier when requested."""
    if not is_bagging_enabled(strategy):
        return estimator

    return BaggingClassifier(
        estimator=estimator,
        n_estimators=bagging_n_estimators,
        max_samples=bagging_max_samples,
        max_features=bagging_max_features,
        bootstrap=bagging_bootstrap,
        bootstrap_features=bagging_bootstrap_features,
        random_state=random_state,
        n_jobs=1,
    )


def adapt_param_grid_for_strategy(
    grid: Dict[str, list],
    *,
    strategy: str,
    pipeline_prefix: str,
) -> Dict[str, list]:
    """Rewrite base-estimator parameter keys for a bagged classifier."""
    if not is_bagging_enabled(strategy):
        return dict(grid)

    adapted: Dict[str, list] = {}
    for key, values in grid.items():
        if key.startswith(pipeline_prefix):
            inner = key[len(pipeline_prefix):]
            adapted[f"{pipeline_prefix}estimator__{inner}"] = values
        else:
            adapted[f"estimator__{key}"] = values
    return adapted
