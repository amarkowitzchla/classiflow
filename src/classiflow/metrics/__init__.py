"""Metrics computation and scoring."""

from classiflow.metrics.binary import compute_binary_metrics
from classiflow.metrics.calibration import compute_probability_quality
from classiflow.metrics.calibration_policy import decide_calibration, CalibrationDecision
from classiflow.metrics.decision import compute_decision_metrics
from classiflow.metrics.probability_quality_checks import (
    ProbQualityRuleResult,
    evaluate_probability_quality_checks,
    build_probability_quality_next_steps,
    collect_probability_quality_plot_payload,
)
from classiflow.metrics.scorers import get_scorers, SCORER_ORDER

__all__ = [
    "compute_binary_metrics",
    "compute_decision_metrics",
    "compute_probability_quality",
    "decide_calibration",
    "CalibrationDecision",
    "ProbQualityRuleResult",
    "evaluate_probability_quality_checks",
    "build_probability_quality_next_steps",
    "collect_probability_quality_plot_payload",
    "get_scorers",
    "SCORER_ORDER",
]
