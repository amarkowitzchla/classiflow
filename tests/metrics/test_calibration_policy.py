"""Unit tests for calibration auto-disable policy decisions."""

from __future__ import annotations

from classiflow.metrics.calibration_policy import decide_calibration


def _base_uncal() -> dict:
    return {
        "accuracy_top1": 0.95,
        "confidence_gap_top1": -0.05,
        "brier_recommended": 0.20,
        "log_loss": 0.40,
        "ece_top1": 0.08,
        "ece_ovr_macro": 0.09,
    }


def _base_cal() -> dict:
    return {
        "brier_recommended": 0.19,
        "log_loss": 0.401,
        "ece_top1": 0.07,
        "ece_ovr_macro": 0.095,
    }


def test_r1_underconfidence_high_accuracy_disables():
    uncal = _base_uncal()
    uncal["confidence_gap_top1"] = -0.20
    decision = decide_calibration(
        mode="meta",
        enabled_requested="auto",
        force_keep=False,
        apply_to_modes=["meta"],
        thresholds={},
        uncal_metrics=uncal,
        cal_metrics=_base_cal(),
        n_samples=500,
        class_counts={"a": 200, "b": 150, "c": 150},
    )
    assert not decision.enabled_final
    assert decision.decision == "disabled_by_policy"
    assert any("R1_underconfidence_guardrail" in reason for reason in decision.reasons)


def test_r2_near_perfect_accuracy_disables():
    uncal = _base_uncal()
    uncal["accuracy_top1"] = 0.99
    decision = decide_calibration(
        mode="meta",
        enabled_requested="auto",
        force_keep=False,
        apply_to_modes=["meta"],
        thresholds={},
        uncal_metrics=uncal,
        cal_metrics=_base_cal(),
        n_samples=500,
        class_counts={"a": 200, "b": 150, "c": 150},
    )
    assert not decision.enabled_final
    assert decision.decision == "disabled_by_policy"
    assert any("R2_near_perfect_accuracy" in reason for reason in decision.reasons)


def test_r3_small_n_disables():
    decision = decide_calibration(
        mode="meta",
        enabled_requested="auto",
        force_keep=False,
        apply_to_modes=["meta"],
        thresholds={},
        uncal_metrics=_base_uncal(),
        cal_metrics=_base_cal(),
        n_samples=120,
        class_counts={"a": 40, "b": 40, "c": 40},
    )
    assert not decision.enabled_final
    assert decision.decision == "disabled_by_policy"
    assert any("R3_min_samples" in reason for reason in decision.reasons)


def test_r4_worse_metrics_disables():
    cal = _base_cal()
    cal["brier_recommended"] = 0.205
    cal["log_loss"] = 0.43
    cal["ece_ovr_macro"] = 0.12
    decision = decide_calibration(
        mode="meta",
        enabled_requested="auto",
        force_keep=False,
        apply_to_modes=["meta"],
        thresholds={},
        uncal_metrics=_base_uncal(),
        cal_metrics=cal,
        n_samples=500,
        class_counts={"a": 200, "b": 150, "c": 150},
    )
    assert not decision.enabled_final
    assert decision.decision == "disabled_by_metrics"
    assert any("R4_failed_improvement" in reason for reason in decision.reasons)


def test_r4_improvement_retains():
    cal = _base_cal()
    cal["brier_recommended"] = 0.195
    cal["log_loss"] = 0.405
    cal["ece_ovr_macro"] = 0.095
    decision = decide_calibration(
        mode="meta",
        enabled_requested="auto",
        force_keep=False,
        apply_to_modes=["meta"],
        thresholds={},
        uncal_metrics=_base_uncal(),
        cal_metrics=cal,
        n_samples=500,
        class_counts={"a": 200, "b": 150, "c": 150},
    )
    assert decision.enabled_final
    assert decision.decision == "retained"
