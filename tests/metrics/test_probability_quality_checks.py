from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from classiflow.metrics.probability_quality_checks import evaluate_probability_quality_checks


def _write_prob_quality_run(
    tmp_path: Path,
    *,
    mode: str = "meta",
    final_variant: str = "uncalibrated",
    uncalibrated: dict | None = None,
    calibrated: dict | None = None,
    n_samples: int = 120,
    bins: int = 10,
    curve_counts: list[int] | None = None,
    class_counts: dict | None = None,
    write_binary_curve: bool = False,
) -> Path:
    run_dir = tmp_path / "technical"
    prefix = "binary"
    if mode == "multiclass":
        prefix = "multiclass"
    if mode == "hierarchical":
        prefix = "hierarchical"
    var_dir = run_dir / "fold1" / f"{prefix}_none"
    var_dir.mkdir(parents=True, exist_ok=True)

    uncal = {
        "brier_recommended": 0.02,
        "log_loss": 0.20,
        "ece_top1": 0.08,
        "ece_ovr_macro": 0.04,
        "accuracy_top1": 0.95,
        "mean_confidence_top1": 0.90,
        "confidence_gap_top1": -0.05,
        "ece_ovr__A": 0.04,
    }
    cal = {
        "brier_recommended": 0.02,
        "log_loss": 0.20,
        "ece_top1": 0.08,
        "ece_ovr_macro": 0.04,
        "accuracy_top1": 0.95,
        "mean_confidence_top1": 0.90,
        "confidence_gap_top1": -0.05,
        "ece_ovr__A": 0.04,
    }
    if uncalibrated:
        uncal.update(uncalibrated)
    if calibrated:
        cal.update(calibrated)

    fold_payload = {
        "uncalibrated": uncal,
        "calibrated": cal,
        "final_variant": final_variant,
        "calibration_decision": {"n_samples": n_samples},
    }
    if class_counts is not None:
        fold_payload["class_counts"] = class_counts

    payload = {
        "artifact_registry": {
            "probability_quality": {
                "final_variant": final_variant,
                "folds": {"fold_1_none": fold_payload},
            }
        },
        "config": {"task_mode": mode},
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    (var_dir / "calibration_metadata.json").write_text(
        json.dumps({"bins": bins}), encoding="utf-8"
    )

    counts = curve_counts or [20] * bins
    curve = pd.DataFrame(
        {
            "bin_id": list(range(len(counts))),
            "n": counts,
            "mean_pred": [0.5] * len(counts),
            "frac_pos": [0.5] * len(counts),
        }
    )
    curve.to_csv(var_dir / "calibration_curve_top1_uncalibrated.csv", index=False)
    curve.to_csv(var_dir / "calibration_curve_top1_calibrated.csv", index=False)
    curve.to_csv(var_dir / "calibration_curve.csv", index=False)
    if write_binary_curve:
        curve.to_csv(var_dir / "calibration_curve_binary_pos_uncalibrated.csv", index=False)
    return run_dir


def test_pq_001_low_occupancy_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        n_samples=100,
        curve_counts=[0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="meta")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-001" in by_id
    assert by_id["PQ-001"].severity == "WARN"
    assert any("run.json" in item["artifact"] for item in by_id["PQ-001"].evidence)


def test_pq_002_underconfidence_warn_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"confidence_gap_top1": -0.25, "accuracy_top1": 0.99, "mean_confidence_top1": 0.74},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="meta")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-002" in by_id
    assert by_id["PQ-002"].severity == "WARN"


def test_pq_003_overconfidence_error_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"confidence_gap_top1": 0.12, "accuracy_top1": 0.80, "mean_confidence_top1": 0.92},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="meta")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-003" in by_id
    assert by_id["PQ-003"].severity == "ERROR"


def test_pq_004_calibration_worsened_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"brier_recommended": 0.02, "log_loss": 0.20, "ece_top1": 0.08, "ece_ovr_macro": 0.04},
        calibrated={"brier_recommended": 0.03, "log_loss": 0.25, "ece_top1": 0.12, "ece_ovr_macro": 0.06},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="meta")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-004" in by_id
    assert by_id["PQ-004"].severity == "WARN"


def test_binary_mode_uses_binary_probability_metrics_and_curve(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        mode="binary",
        uncalibrated={
            "brier_binary": 0.11,
            "log_loss": 0.31,
            "ece_binary_pos": 0.08,
            "confidence_gap_top1": 0.07,
            "accuracy_top1": 0.82,
            "mean_confidence_top1": 0.89,
        },
        calibrated={
            "brier_binary": 0.14,
            "log_loss": 0.36,
            "ece_binary_pos": 0.12,
        },
        write_binary_curve=True,
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="binary")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-001" in by_id
    assert "binary_pos" in by_id["PQ-001"].summary
    assert "PQ-004" in by_id
    assert "delta_brier_binary_cal_minus_uncal" in by_id["PQ-004"].measured
    assert "delta_ece_binary_pos_cal_minus_uncal" in by_id["PQ-004"].measured
    assert "delta_brier_recommended_cal_minus_uncal" not in by_id["PQ-004"].measured


def test_pq_005_distribution_shift_triggers_with_independent_metrics(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"brier_recommended": 0.02, "log_loss": 0.20},
    )
    results = evaluate_probability_quality_checks(
        run_dir=run_dir,
        mode="meta",
        independent_metrics={"brier_recommended": 0.05, "log_loss": 0.30},
    )
    by_id = {r.rule_id: r for r in results}
    assert "PQ-005" in by_id


def test_pq_006_weak_class_ovr_warn_when_support_available(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"ece_ovr__A": 0.20, "ece_ovr__B": 0.10},
        class_counts={"A": 40, "B": 40},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="multiclass")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-006" in by_id
    assert by_id["PQ-006"].severity == "WARN"


def test_pq_006_weak_class_ovr_info_with_low_power_caveat(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        mode="multiclass",
        uncalibrated={"ece_ovr__A": 0.22, "ece_ovr__B": 0.18},
        class_counts=None,
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="multiclass")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-006" in by_id
    assert by_id["PQ-006"].severity == "INFO"
    assert "low-power" in by_id["PQ-006"].summary


def test_pq_h001_hierarchical_mismatch_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        mode="hierarchical",
        uncalibrated={"pred_alignment_mismatch_rate": 0.08},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="hierarchical")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-H001" in by_id
    assert by_id["PQ-H001"].severity == "WARN"


def test_pq_007_near_perfect_high_ece_triggers(tmp_path: Path) -> None:
    run_dir = _write_prob_quality_run(
        tmp_path,
        uncalibrated={"accuracy_top1": 0.98, "ece_top1": 0.20},
    )
    results = evaluate_probability_quality_checks(run_dir=run_dir, mode="meta")
    by_id = {r.rule_id: r for r in results}
    assert "PQ-007" in by_id
    assert by_id["PQ-007"].severity == "INFO"
