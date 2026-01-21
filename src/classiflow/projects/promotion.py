"""Promotion gate evaluation and decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from classiflow.projects.project_models import ThresholdsConfig


METRIC_ALIASES = {
    "sensitivity": "recall",
    "tpr": "recall",
    "balanced_acc": "balanced_accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1": "f1_macro",  # Map f1 to f1_macro for multi-class problems
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "roc_auc_macro": "roc_auc_ovr_macro",
    "roc_auc_ovr_macro": "roc_auc_ovr_macro",
}


def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(key, key)


@dataclass
class GateResult:
    """Result of evaluating a gate."""

    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]


def _evaluate_required(metrics: Dict[str, float], required: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons = []
    ok = True
    for metric, threshold in required.items():
        key = normalize_metric_name(metric)
        value = metrics.get(key)
        if value is None or np.isnan(value):
            ok = False
            reasons.append(f"Missing metric: {metric}")
            continue
        if value < threshold:
            ok = False
            reasons.append(f"{metric}={value:.4f} < {threshold:.4f}")
    return ok, reasons


def _evaluate_safety(metrics: Dict[str, float], safety: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons = []
    ok = True
    for metric, max_allowed in safety.items():
        key = normalize_metric_name(metric)
        value = metrics.get(key)
        if value is None or np.isnan(value):
            ok = False
            reasons.append(f"Missing safety metric: {metric}")
            continue
        if value > max_allowed:
            ok = False
            reasons.append(f"{metric}={value:.4f} > {max_allowed:.4f}")
    return ok, reasons


def _evaluate_stability(
    per_fold: Dict[str, List[float]],
    std_max: Dict[str, float],
    pass_rate_min: float,
    required: Dict[str, float],
) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons = []
    ok = True
    stability_metrics = {}
    catastrophic_metrics = {"recall", "sensitivity", "tpr"}
    for metric, values in per_fold.items():
        norm_metric = normalize_metric_name(metric)
        values = [v for v in values if v == v]
        if not values:
            ok = False
            reasons.append(f"No fold values for {metric}")
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stability_metrics[f"{norm_metric}_std"] = std
        std_thresh = std_max.get(metric) or std_max.get(norm_metric)
        if std_thresh is not None and std > std_thresh:
            ok = False
            reasons.append(f"{metric} std={std:.4f} > {std_thresh:.4f}")
        if metric in required or norm_metric in required:
            threshold = required.get(metric, required.get(norm_metric))
            pass_rate = float(np.mean([v >= threshold for v in values]))
            stability_metrics[f"{norm_metric}_pass_rate"] = pass_rate
            if pass_rate < pass_rate_min:
                ok = False
                reasons.append(
                    f"{metric} pass_rate={pass_rate:.2f} < {pass_rate_min:.2f}"
                )
        if norm_metric in catastrophic_metrics and any(v <= 0 for v in values):
            ok = False
            reasons.append(f"{metric} catastrophic failure (value <= 0)")
    return ok, reasons, stability_metrics


def _evaluate_calibration(
    metrics: Dict[str, float],
    brier_max: float,
    ece_max: float,
) -> Tuple[bool, List[str]]:
    reasons = []
    ok = True
    brier = metrics.get("brier_calibrated")
    if brier is None or np.isnan(brier):
        ok = False
        reasons.append("Missing calibrated metric: brier_calibrated")
    elif brier > brier_max:
        ok = False
        reasons.append(f"brier_calibrated={brier:.4f} > {brier_max:.4f}")
    ece = metrics.get("ece_calibrated")
    if ece is None or np.isnan(ece):
        ok = False
        reasons.append("Missing calibrated metric: ece_calibrated")
    elif ece > ece_max:
        ok = False
        reasons.append(f"ece_calibrated={ece:.4f} > {ece_max:.4f}")
    return ok, reasons


def evaluate_promotion(
    thresholds: ThresholdsConfig,
    technical_metrics: Dict[str, float],
    technical_per_fold: Dict[str, List[float]],
    test_metrics: Dict[str, float],
) -> Dict[str, GateResult]:
    """Evaluate promotion gates for technical validation and independent test."""
    results: Dict[str, GateResult] = {}

    tech_req_ok, tech_req_reasons = _evaluate_required(
        technical_metrics, thresholds.technical_validation.required
    )
    tech_safety_ok, tech_safety_reasons = _evaluate_safety(
        technical_metrics, thresholds.technical_validation.safety
    )
    cal_thresh = thresholds.promotion.calibration
    tech_cal_ok, tech_cal_reasons = _evaluate_calibration(
        technical_metrics,
        cal_thresh.brier_max,
        cal_thresh.ece_max,
    )

    stability_ok = True
    stability_reasons: List[str] = []
    stability_metrics: Dict[str, float] = {}
    if thresholds.technical_validation.stability:
        stability_ok, stability_reasons, stability_metrics = _evaluate_stability(
            technical_per_fold,
            thresholds.technical_validation.stability.std_max,
            thresholds.technical_validation.stability.pass_rate_min,
            thresholds.technical_validation.required,
        )

    tech_ok = tech_req_ok and tech_safety_ok and stability_ok and tech_cal_ok
    tech_reasons = tech_req_reasons + tech_safety_reasons + stability_reasons + tech_cal_reasons
    results["technical_validation"] = GateResult(
        passed=tech_ok,
        reasons=tech_reasons,
        metrics={**technical_metrics, **stability_metrics},
    )

    test_req_ok, test_req_reasons = _evaluate_required(
        test_metrics, thresholds.independent_test.required
    )
    test_safety_ok, test_safety_reasons = _evaluate_safety(
        test_metrics, thresholds.independent_test.safety
    )
    test_cal_ok, test_cal_reasons = _evaluate_calibration(
        test_metrics,
        cal_thresh.brier_max,
        cal_thresh.ece_max,
    )
    test_ok = test_req_ok and test_safety_ok and test_cal_ok
    test_reasons = test_req_reasons + test_safety_reasons + test_cal_reasons
    results["independent_test"] = GateResult(
        passed=test_ok,
        reasons=test_reasons,
        metrics=test_metrics,
    )

    return results


def promotion_decision(results: Dict[str, GateResult]) -> Tuple[bool, List[str]]:
    """Summarize overall promotion decision."""
    failed = [phase for phase, res in results.items() if not res.passed]
    if not failed:
        return True, []
    reasons = []
    for phase in failed:
        for reason in results[phase].reasons:
            reasons.append(f"{phase}: {reason}")
    return False, reasons
