"""Promotion gate evaluation and decision logic."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import operator

import numpy as np

from classiflow.projects.project_models import PromotionGateSpec, ThresholdsConfig
from classiflow.projects.promotion_templates import (
    TEMPLATE_DEFAULT_F1_BALACC,
    get_promotion_gate_template,
)


METRIC_ALIASES = {
    "sensitivity": "recall",
    "tpr": "recall",
    "balanced_acc": "balanced_accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_score": "f1_macro",
    "f1_macro": "f1_macro",
    "f1": "f1_macro",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "roc_auc_macro": "roc_auc_ovr_macro",
    "roc_auc_ovr_macro": "roc_auc_ovr_macro",
    "precision": "precision",
    "specificity": "specificity",
    "mcc": "mcc",
    "recall": "recall",
}

_COMPARATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}


def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(key, key)


def _metric_candidates(name: str) -> List[str]:
    normalized = normalize_metric_name(name)
    candidates = [normalized]
    if normalized == "f1_macro":
        candidates.extend(["f1_weighted", "f1"])
    elif normalized == "f1":
        candidates.extend(["f1_macro", "f1_weighted"])
    elif normalized == "roc_auc":
        candidates.extend(["roc_auc_ovr_macro", "roc_auc_macro"])
    elif normalized == "recall":
        candidates.extend(["sensitivity", "tpr"])
    return candidates


def resolve_metric(metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    for key in _metric_candidates(metric_name):
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _resolve_per_fold_values(per_fold: Dict[str, List[float]], metric_name: str) -> Optional[List[float]]:
    for key in _metric_candidates(metric_name):
        values = per_fold.get(key)
        if values:
            return values
    return None


@dataclass
class PerGateResult:
    """Result of evaluating one gate."""

    phase: str
    metric: str
    op: str
    threshold: float
    observed_value: Optional[float]
    passed: bool
    scope: str
    aggregation: str
    notes: Optional[str] = None


@dataclass
class GateResult:
    """Result of evaluating a phase."""

    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]
    per_gate_results: List[PerGateResult]
    template: Dict[str, object]


def _compare(value: float, op: str, threshold: float) -> bool:
    return _COMPARATORS[op](value, threshold)


def _aggregate_gate_value(
    metric_name: str,
    aggregation: str,
    summary_metrics: Dict[str, float],
    per_fold: Dict[str, List[float]],
) -> Optional[float]:
    aggregation = aggregation.lower()
    if aggregation == "mean":
        value = resolve_metric(summary_metrics, metric_name)
        if value is not None and not np.isnan(value):
            return float(value)

    values = _resolve_per_fold_values(per_fold, metric_name)
    if values is None:
        # Median/min/percentile require fold-level values; mean can fallback to summary.
        return None
    values = [float(v) for v in values if v == v]
    if not values:
        return None

    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "min":
        return float(np.min(values))
    if aggregation.startswith("p"):
        percentile = int(aggregation[1:])
        return float(np.percentile(values, percentile))
    return None


def _phase_name_for_scope(scope: str) -> List[str]:
    if scope == "outer":
        return ["technical_validation"]
    if scope == "independent":
        return ["independent_test"]
    return ["technical_validation", "independent_test"]


def _legacy_gates(thresholds: ThresholdsConfig) -> List[PromotionGateSpec]:
    gates: List[PromotionGateSpec] = []
    for metric, threshold in thresholds.technical_validation.required.items():
        gates.append(
            PromotionGateSpec(metric=metric, op=">=", threshold=float(threshold), scope="outer", aggregation="mean")
        )
    for metric, threshold in thresholds.independent_test.required.items():
        gates.append(
            PromotionGateSpec(metric=metric, op=">=", threshold=float(threshold), scope="independent", aggregation="mean")
        )
    for metric, threshold in thresholds.technical_validation.safety.items():
        gates.append(
            PromotionGateSpec(metric=metric, op="<=", threshold=float(threshold), scope="outer", aggregation="mean")
        )
    for metric, threshold in thresholds.independent_test.safety.items():
        gates.append(
            PromotionGateSpec(metric=metric, op="<=", threshold=float(threshold), scope="independent", aggregation="mean")
        )
    return gates


def resolve_promotion_gates(thresholds: ThresholdsConfig) -> Dict[str, object]:
    """Resolve manual/template/default gates with precedence and provenance."""
    manual_gates = list(thresholds.promotion_gates)
    template_id = thresholds.promotion_gate_template
    ignored_template: Optional[Dict[str, object]] = None

    if manual_gates:
        if template_id:
            template = get_promotion_gate_template(template_id)
            ignored_template = {
                "template_id": template.template_id,
                "display_name": template.display_name,
                "version": template.version,
                "status": "ignored_due_to_manual_override",
            }
        return {
            "source": "manual",
            "template_id": "manual_override",
            "display_name": "Manual Promotion Gates",
            "description": "Gates from thresholds.promotion_gates",
            "layman_explanation": "Custom promotion gates defined by the project.",
            "version": 1,
            "gates": manual_gates,
            "ignored_template": ignored_template,
        }

    if template_id:
        template = get_promotion_gate_template(template_id)
        return {
            "source": "template",
            "template_id": template.template_id,
            "display_name": template.display_name,
            "description": template.description,
            "layman_explanation": template.layman_explanation,
            "version": template.version,
            "gates": list(template.gates),
            "ignored_template": None,
        }

    legacy = _legacy_gates(thresholds)
    if legacy:
        return {
            "source": "legacy",
            "template_id": "legacy_thresholds",
            "display_name": "Legacy Threshold Gates",
            "description": "Derived from technical/independent required and safety threshold maps.",
            "layman_explanation": "Legacy threshold configuration was used for promotion checks.",
            "version": 1,
            "gates": legacy,
            "ignored_template": None,
        }

    template = get_promotion_gate_template(TEMPLATE_DEFAULT_F1_BALACC)
    return {
        "source": "default",
        "template_id": template.template_id,
        "display_name": template.display_name,
        "description": template.description,
        "layman_explanation": template.layman_explanation,
        "version": template.version,
        "gates": list(template.gates),
        "ignored_template": None,
    }


def _evaluate_calibration(
    metrics: Dict[str, float],
    brier_max: Optional[float],
    ece_max: Optional[float],
) -> Tuple[bool, List[str], List[PerGateResult]]:
    reasons = []
    ok = True
    per_gate: List[PerGateResult] = []
    if brier_max is not None:
        brier = metrics.get("brier_calibrated")
        passed = brier is not None and not np.isnan(brier) and brier <= brier_max
        if not passed:
            ok = False
            if brier is None or np.isnan(brier):
                reasons.append("Missing calibrated metric: brier_calibrated")
            else:
                reasons.append(f"brier_calibrated={brier:.4f} > {brier_max:.4f}")
        per_gate.append(
            PerGateResult(
                phase="",
                metric="brier_calibrated",
                op="<=",
                threshold=float(brier_max),
                observed_value=float(brier) if brier is not None and not np.isnan(brier) else None,
                passed=passed,
                scope="both",
                aggregation="mean",
            )
        )
    if ece_max is not None:
        ece = metrics.get("ece_calibrated")
        passed = ece is not None and not np.isnan(ece) and ece <= ece_max
        if not passed:
            ok = False
            if ece is None or np.isnan(ece):
                reasons.append("Missing calibrated metric: ece_calibrated")
            else:
                reasons.append(f"ece_calibrated={ece:.4f} > {ece_max:.4f}")
        per_gate.append(
            PerGateResult(
                phase="",
                metric="ece_calibrated",
                op="<=",
                threshold=float(ece_max),
                observed_value=float(ece) if ece is not None and not np.isnan(ece) else None,
                passed=passed,
                scope="both",
                aggregation="mean",
            )
        )
    return ok, reasons, per_gate


def evaluate_promotion(
    thresholds: ThresholdsConfig,
    technical_metrics: Dict[str, float],
    technical_per_fold: Dict[str, List[float]],
    test_metrics: Dict[str, float],
) -> Dict[str, GateResult]:
    """Evaluate promotion gates for technical validation and independent test."""
    resolved = resolve_promotion_gates(thresholds)
    gates: List[PromotionGateSpec] = resolved["gates"]

    phase_inputs = {
        "technical_validation": {
            "summary": technical_metrics,
            "per_fold": technical_per_fold,
        },
        "independent_test": {
            "summary": test_metrics,
            "per_fold": {},
        },
    }

    phase_results: Dict[str, GateResult] = {
        "technical_validation": GateResult(
            passed=True,
            reasons=[],
            metrics=dict(technical_metrics),
            per_gate_results=[],
            template={k: v for k, v in resolved.items() if k != "gates"},
        ),
        "independent_test": GateResult(
            passed=True,
            reasons=[],
            metrics=dict(test_metrics),
            per_gate_results=[],
            template={k: v for k, v in resolved.items() if k != "gates"},
        ),
    }

    for gate in gates:
        for phase in _phase_name_for_scope(gate.scope):
            observed = _aggregate_gate_value(
                metric_name=gate.metric,
                aggregation=gate.aggregation,
                summary_metrics=phase_inputs[phase]["summary"],
                per_fold=phase_inputs[phase]["per_fold"],
            )
            passed = observed is not None and _compare(observed, gate.op, gate.threshold)
            per_gate_result = PerGateResult(
                phase=phase,
                metric=gate.metric,
                op=gate.op,
                threshold=float(gate.threshold),
                observed_value=observed,
                passed=passed,
                scope=gate.scope,
                aggregation=gate.aggregation,
                notes=gate.notes,
            )
            phase_results[phase].per_gate_results.append(per_gate_result)
            if not passed:
                phase_results[phase].passed = False
                if observed is None:
                    phase_results[phase].reasons.append(
                        f"Missing metric: {gate.metric} ({gate.aggregation}, scope={gate.scope})"
                    )
                else:
                    phase_results[phase].reasons.append(
                        f"{gate.metric}={observed:.4f} {gate.op} {gate.threshold:.4f} failed"
                    )

    cal_thresh = thresholds.promotion.calibration
    for phase in ["technical_validation", "independent_test"]:
        cal_ok, cal_reasons, cal_results = _evaluate_calibration(
            phase_inputs[phase]["summary"],
            cal_thresh.brier_max,
            cal_thresh.ece_max,
        )
        for cal_result in cal_results:
            cal_result.phase = phase
        phase_results[phase].per_gate_results.extend(cal_results)
        if not cal_ok:
            phase_results[phase].passed = False
            phase_results[phase].reasons.extend(cal_reasons)

    return phase_results


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


def per_gate_rows(results: Dict[str, GateResult]) -> List[Dict[str, object]]:
    """Convert per-gate results to report/JSON rows."""
    rows: List[Dict[str, object]] = []
    for phase, result in results.items():
        for gate in result.per_gate_results:
            payload = asdict(gate)
            payload["phase"] = phase
            rows.append(payload)
    return rows
