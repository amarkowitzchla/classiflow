from __future__ import annotations

from pathlib import Path

import pandas as pd

from classiflow.metrics.binary_learners import build_binary_learner_warnings


def test_binary_learner_warning_rules_trigger_expected_severities(tmp_path: Path) -> None:
    by_fold = pd.DataFrame(
        [
            {
                "fold": 1,
                "sampler": "none",
                "class_name": "A",
                "roc_auc_ovr": 0.55,
                "n_pos": 40,
                "p_std": 0.004,
                "predicted_positive_rate_0_5": 0.999,
            },
            {
                "fold": 1,
                "sampler": "none",
                "class_name": "B",
                "roc_auc_ovr": 0.55,
                "n_pos": 10,
                "p_std": 0.08,
                "predicted_positive_rate_0_5": 0.40,
            },
            {
                "fold": 1,
                "sampler": "none",
                "class_name": "C",
                "roc_auc_ovr": 0.98,
                "n_pos": 35,
                "p_std": 0.05,
                "predicted_positive_rate_0_5": 0.45,
            },
            {
                "fold": 2,
                "sampler": "none",
                "class_name": "C",
                "roc_auc_ovr": 0.70,
                "n_pos": 38,
                "p_std": 0.05,
                "predicted_positive_rate_0_5": 0.50,
            },
            {
                "fold": 1,
                "sampler": "none",
                "class_name": "D",
                "roc_auc_ovr": 0.999,
                "n_pos": 32,
                "p_std": 0.06,
                "predicted_positive_rate_0_5": 0.50,
            },
        ]
    )
    summary = pd.DataFrame([{"class_name": c} for c in ["A", "B", "C", "D"]])
    manifest_rows = [
        {
            "fold": 1,
            "sampler": "none",
            "base_ovr_proba_path": "fold1/binary_none/base_ovr_proba_fold1.npz",
        },
        {
            "fold": 2,
            "sampler": "none",
            "base_ovr_proba_path": "fold2/binary_none/base_ovr_proba_fold2.npz",
        },
    ]

    warnings = build_binary_learner_warnings(
        by_fold_df=by_fold,
        summary_df=summary,
        run_dir=tmp_path,
        manifest_rows=manifest_rows,
        classes=["A", "B", "C", "D"],
    )
    index = {(w["rule_id"], w["class_name"], w["severity"]) for w in warnings}

    assert ("BL-001", "A", "ERROR") in index
    assert ("BL-002", "A", "WARN") in index
    assert ("BL-003", "A", "WARN") in index
    assert ("BL-003", "B", "INFO") in index
    assert ("BL-004", "C", "WARN") in index
    assert ("BL-005", "D", "WARN") in index
    assert ("BL-006", "B", "INFO") in index
