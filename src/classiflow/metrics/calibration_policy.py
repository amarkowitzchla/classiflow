"""Policy engine for deciding whether calibrated probabilities should be kept."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional
import math


DEFAULT_POLICY_THRESHOLDS: Dict[str, float] = {
    "underconfidence_gap": -0.10,
    "high_accuracy": 0.90,
    "near_perfect_accuracy": 0.97,
    "min_calibration_n": 200,
    "min_class_n": 25,
    "min_brier_improvement": 0.002,
    "max_log_loss_regression": 0.01,
    "max_ece_ovr_regression": 0.01,
}


@dataclass
class CalibrationDecision:
    enabled_final: bool
    enabled_requested: str
    decision: str
    reasons: List[str]
    comparisons: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _comparison(
    metric: str,
    uncal_metrics: Mapping[str, Any],
    cal_metrics: Mapping[str, Any],
) -> Optional[Dict[str, float]]:
    uncal = _as_float(uncal_metrics.get(metric))
    cal = _as_float(cal_metrics.get(metric))
    if uncal is None or cal is None:
        return None
    return {"uncal": uncal, "cal": cal, "delta": cal - uncal}


def decide_calibration(
    *,
    mode: str,
    enabled_requested: str,
    force_keep: bool,
    apply_to_modes: List[str],
    thresholds: Mapping[str, Any],
    uncal_metrics: Mapping[str, Any],
    cal_metrics: Optional[Mapping[str, Any]],
    n_samples: int,
    class_counts: Optional[Mapping[str, int]] = None,
) -> CalibrationDecision:
    """
    Decide if calibration should be retained for final predictions.

    Decision labels:
    - retained
    - disabled_by_user
    - disabled_by_policy
    - disabled_by_metrics
    """
    thresholds_used = dict(DEFAULT_POLICY_THRESHOLDS)
    thresholds_used.update({k: v for k, v in thresholds.items() if v is not None})
    reasons: List[str] = []

    comparisons: Dict[str, Dict[str, float]] = {}
    if cal_metrics is not None:
        for metric in ("brier_recommended", "log_loss", "ece_top1", "ece_ovr_macro"):
            comp = _comparison(metric, uncal_metrics, cal_metrics)
            if comp is not None:
                comparisons[metric] = comp

    if enabled_requested == "false":
        return CalibrationDecision(
            enabled_final=False,
            enabled_requested=enabled_requested,
            decision="disabled_by_user",
            reasons=["Calibration disabled by user configuration (enabled=false)."],
            comparisons=comparisons,
        )

    policy_applies = enabled_requested == "auto" and mode in set(apply_to_modes)

    # R1-R3 apply only in auto mode for selected modes.
    if policy_applies:
        acc_top1 = _as_float(uncal_metrics.get("accuracy_top1"))
        gap = _as_float(uncal_metrics.get("confidence_gap_top1"))
        if (
            gap is not None
            and acc_top1 is not None
            and gap <= float(thresholds_used["underconfidence_gap"])
            and acc_top1 >= float(thresholds_used["high_accuracy"])
        ):
            reasons.append(
                "R1_underconfidence_guardrail:"
                f" gap={gap:.6f} <= {float(thresholds_used['underconfidence_gap']):.6f} AND"
                f" acc_top1={acc_top1:.6f} >= {float(thresholds_used['high_accuracy']):.6f}"
            )

        if (
            acc_top1 is not None
            and acc_top1 >= float(thresholds_used["near_perfect_accuracy"])
        ):
            reasons.append(
                "R2_near_perfect_accuracy:"
                f" acc_top1={acc_top1:.6f} >= {float(thresholds_used['near_perfect_accuracy']):.6f}"
            )

        if n_samples < int(thresholds_used["min_calibration_n"]):
            reasons.append(
                "R3_min_samples:"
                f" n_samples={int(n_samples)} < {int(thresholds_used['min_calibration_n'])}"
            )

        if class_counts:
            min_class_n = min(int(v) for v in class_counts.values())
            if min_class_n < int(thresholds_used["min_class_n"]):
                reasons.append(
                    "R3_min_class_samples:"
                    f" min_class_n={min_class_n} < {int(thresholds_used['min_class_n'])}"
                )

        if reasons:
            return CalibrationDecision(
                enabled_final=False,
                enabled_requested=enabled_requested,
                decision="disabled_by_policy",
                reasons=reasons,
                comparisons=comparisons,
            )

    if force_keep and enabled_requested == "true":
        return CalibrationDecision(
            enabled_final=True,
            enabled_requested=enabled_requested,
            decision="retained",
            reasons=["Force-keep enabled: skipping R4 metric regression guardrail."],
            comparisons=comparisons,
        )

    # In auto mode for non-target modes, keep existing behavior (retain if calibration exists).
    if enabled_requested == "auto" and not policy_applies:
        if cal_metrics is None:
            return CalibrationDecision(
                enabled_final=False,
                enabled_requested=enabled_requested,
                decision="disabled_by_metrics",
                reasons=["Calibration unavailable for this fold; using uncalibrated probabilities."],
                comparisons=comparisons,
            )
        return CalibrationDecision(
            enabled_final=True,
            enabled_requested=enabled_requested,
            decision="retained",
            reasons=[f"Auto policy not applied for mode={mode}; retaining existing calibration behavior."],
            comparisons=comparisons,
        )

    # R4 - calibration must help; applies to auto (post R1-R3) and true (unless force_keep).
    if cal_metrics is None:
        return CalibrationDecision(
            enabled_final=False,
            enabled_requested=enabled_requested,
            decision="disabled_by_metrics",
            reasons=["R4_failed_improvement: calibrated metrics unavailable."],
            comparisons=comparisons,
        )

    brier_uncal = _as_float(uncal_metrics.get("brier_recommended"))
    brier_cal = _as_float(cal_metrics.get("brier_recommended"))
    ll_uncal = _as_float(uncal_metrics.get("log_loss"))
    ll_cal = _as_float(cal_metrics.get("log_loss"))
    ece_ovr_uncal = _as_float(uncal_metrics.get("ece_ovr_macro"))
    ece_ovr_cal = _as_float(cal_metrics.get("ece_ovr_macro"))

    keep = True
    if brier_uncal is None or brier_cal is None:
        keep = False
        reasons.append("R4_failed_improvement: missing brier_recommended comparison.")
    else:
        min_improvement = float(thresholds_used["min_brier_improvement"])
        if brier_cal > (brier_uncal - min_improvement):
            keep = False
            reasons.append(
                "R4_failed_improvement:"
                f" brier_recommended delta={brier_cal - brier_uncal:+.6f},"
                f" required <= {-min_improvement:.6f}"
            )

    if ll_uncal is None or ll_cal is None:
        keep = False
        reasons.append("R4_failed_improvement: missing log_loss comparison.")
    else:
        max_ll_reg = float(thresholds_used["max_log_loss_regression"])
        if ll_cal > (ll_uncal + max_ll_reg):
            keep = False
            reasons.append(
                "R4_failed_improvement:"
                f" log_loss delta={ll_cal - ll_uncal:+.6f},"
                f" allowed <= {max_ll_reg:.6f}"
            )

    if ece_ovr_uncal is not None and ece_ovr_cal is not None:
        max_ece_reg = float(thresholds_used["max_ece_ovr_regression"])
        if ece_ovr_cal > (ece_ovr_uncal + max_ece_reg):
            keep = False
            reasons.append(
                "R4_failed_improvement:"
                f" ece_ovr_macro delta={ece_ovr_cal - ece_ovr_uncal:+.6f},"
                f" allowed <= {max_ece_reg:.6f}"
            )

    if keep:
        return CalibrationDecision(
            enabled_final=True,
            enabled_requested=enabled_requested,
            decision="retained",
            reasons=["R4_passed: calibration improved probability quality within guardrails."],
            comparisons=comparisons,
        )

    return CalibrationDecision(
        enabled_final=False,
        enabled_requested=enabled_requested,
        decision="disabled_by_metrics",
        reasons=reasons,
        comparisons=comparisons,
    )
