from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from classiflow.projects.reporting import write_technical_report


def _seed_probability_quality_artifacts(run_dir: Path, *, mode: str = "meta") -> None:
    prefix = "binary"
    if mode == "multiclass":
        prefix = "multiclass"
    if mode == "hierarchical":
        prefix = "hierarchical"
    var_dir = run_dir / "fold1" / f"{prefix}_none"
    var_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_registry": {
            "probability_quality": {
                "final_variant": "uncalibrated",
                "folds": {
                    "fold_1_none": {
                        "uncalibrated": {
                            "brier_recommended": 0.02,
                            "log_loss": 0.20,
                            "ece_top1": 0.18,
                            "ece_ovr_macro": 0.05,
                            "accuracy_top1": 0.98,
                            "mean_confidence_top1": 0.70,
                            "confidence_gap_top1": -0.28,
                            "pred_alignment_mismatch_rate": 0.06 if mode == "hierarchical" else 0.0,
                        },
                        "calibrated": {
                            "brier_recommended": 0.03,
                            "log_loss": 0.25,
                            "ece_top1": 0.22,
                            "ece_ovr_macro": 0.06,
                            "accuracy_top1": 0.98,
                            "mean_confidence_top1": 0.76,
                            "confidence_gap_top1": -0.22,
                        },
                        "final_variant": "uncalibrated",
                        "calibration_decision": {"n_samples": 90},
                    }
                },
            }
        }
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    (var_dir / "calibration_metadata.json").write_text(json.dumps({"bins": 10}), encoding="utf-8")
    curve = pd.DataFrame(
        {
            "bin_id": list(range(10)),
            "n": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "mean_pred": [0.5] * 10,
            "frac_pos": [0.5] * 10,
        }
    )
    curve.to_csv(var_dir / "calibration_curve_top1_uncalibrated.csv", index=False)
    curve.to_csv(var_dir / "calibration_curve.csv", index=False)


def test_write_technical_report_includes_probability_quality_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "technical"
    reports_dir = run_dir / "reports"
    _seed_probability_quality_artifacts(run_dir)

    report_path = write_technical_report(
        reports_dir,
        metrics_summary={"f1_macro": 0.9},
        per_fold={"f1_macro": [0.9, 0.91, 0.89]},
        notes=[],
        run_dir=run_dir,
        task_mode="meta",
        promotion_gate_metrics=["ece_top1"],
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Probability Quality Checks (ECE/Brier)" in text
    assert "PQ-001" in text
    assert "PQ-002" in text
    assert "Diagnostic plots generated:" in text
    assert "probability_quality_plots/prob_quality_reliability_top1.png" in text
    assert "probability_quality_plots/prob_quality_confidence_gap.png" in text
    assert "probability_quality_plots/prob_quality_calibration_deltas.png" in text
    assert (reports_dir / "probability_quality_plots" / "prob_quality_reliability_top1.png").exists()
    assert (reports_dir / "probability_quality_plots" / "prob_quality_confidence_gap.png").exists()
    assert (reports_dir / "probability_quality_plots" / "prob_quality_calibration_deltas.png").exists()
    assert "Promotion gate impact: None" in text
    assert "WARN: Promotion gates reference calibration metrics" in text


def test_write_technical_report_probability_quality_renders_for_all_modes(tmp_path: Path) -> None:
    for mode in ("binary", "multiclass", "hierarchical"):
        run_dir = tmp_path / f"technical_{mode}"
        reports_dir = run_dir / "reports"
        _seed_probability_quality_artifacts(run_dir, mode=mode)
        report_path = write_technical_report(
            reports_dir,
            metrics_summary={"f1_macro": 0.9},
            per_fold={"f1_macro": [0.9, 0.91, 0.89]},
            notes=[],
            run_dir=run_dir,
            task_mode=mode,
            promotion_gate_metrics=["ece_top1"],
        )
        text = report_path.read_text(encoding="utf-8")
        assert "## Probability Quality Checks (ECE/Brier)" in text
        assert "Evidence:" in text
        assert "::" in text
        if mode == "hierarchical":
            assert "PQ-H001" in text
