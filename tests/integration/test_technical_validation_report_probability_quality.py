from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from classiflow.projects.reporting import write_technical_report


def _build_fixture_run(run_dir: Path) -> None:
    var_dir = run_dir / "fold1" / "binary_none"
    var_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
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
                                },
                                "calibrated": {
                                    "brier_recommended": 0.03,
                                    "log_loss": 0.24,
                                    "ece_top1": 0.21,
                                    "ece_ovr_macro": 0.07,
                                    "accuracy_top1": 0.98,
                                    "mean_confidence_top1": 0.77,
                                    "confidence_gap_top1": -0.21,
                                },
                                "final_variant": "uncalibrated",
                                "calibration_decision": {"n_samples": 80},
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (var_dir / "calibration_metadata.json").write_text(json.dumps({"bins": 10}), encoding="utf-8")
    pd.DataFrame(
        {
            "bin_id": list(range(10)),
            "n": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "mean_pred": [0.5] * 10,
            "frac_pos": [0.5] * 10,
        }
    ).to_csv(var_dir / "calibration_curve_top1_uncalibrated.csv", index=False)
    pd.DataFrame(
        {
            "bin_id": list(range(10)),
            "n": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "mean_pred": [0.5] * 10,
            "frac_pos": [0.5] * 10,
        }
    ).to_csv(var_dir / "calibration_curve.csv", index=False)

    pd.DataFrame(
        [
            {
                "class_name": "A",
                "n_folds": 1,
                "roc_auc_ovr_mean": 0.82,
                "roc_auc_ovr_std": 0.00,
                "predicted_positive_rate_0_5_mean": 0.12,
                "p_std_mean": 0.08,
                "n_pos_mean": 30.0,
                "health_status": "WARN",
            }
        ]
    ).to_csv(run_dir / "binary_learners_metrics_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": 1,
                "sampler": "none",
                "class_name": "A",
                "n_pos": 30,
                "roc_auc_ovr": 0.82,
                "predicted_positive_rate_0_5": 0.12,
                "p_std": 0.08,
            }
        ]
    ).to_csv(run_dir / "binary_learners_metrics_by_fold.csv", index=False)
    (run_dir / "binary_learners_warnings.json").write_text(
        json.dumps(
            {
                "warnings": [
                    {
                        "severity": "WARN",
                        "rule_id": "BL-004",
                        "class_name": "A",
                        "folds": ["fold1:none"],
                        "finding": "High AUC variance across folds",
                        "recommended_action": "Investigate stability.",
                        "evidence_paths": [
                            "binary_learners_metrics_by_fold.csv",
                            "binary_learners_metrics_summary.csv",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "binary_learners_manifest.json").write_text(
        json.dumps({"classes": ["A"]}),
        encoding="utf-8",
    )
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "binary_ovr_roc_A.png").write_bytes(b"fake")


def test_technical_validation_report_includes_probability_quality_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "technical_run"
    _build_fixture_run(run_dir)
    report_path = write_technical_report(
        run_dir / "reports",
        metrics_summary={"f1_macro": 0.92},
        per_fold={"f1_macro": [0.90, 0.93, 0.93]},
        notes=[],
        run_dir=run_dir,
        task_mode="meta",
    )
    text = report_path.read_text(encoding="utf-8")
    assert "Probability Quality Checks (ECE/Brier)" in text
    assert "Binary Learner Health Report (Meta Under-the-Hood)" in text
    assert "PQ-001" in text
    assert "PQ-004" in text
    assert "probability_quality_plots/prob_quality_reliability_top1.png" in text
    assert "../plots/binary_ovr_roc_A.png" in text
    assert (
        run_dir / "reports" / "probability_quality_plots" / "prob_quality_reliability_top1.png"
    ).exists()
