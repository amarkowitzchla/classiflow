"""Unit tests for promotion gate templates and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classiflow.projects.project_models import PromotionGateSpec, ThresholdsConfig
from classiflow.projects.promotion import evaluate_promotion, per_gate_rows
from classiflow.projects.promotion_templates import list_promotion_gate_templates
from classiflow.projects.reporting import write_promotion_report


def test_template_registry_contains_expected_templates() -> None:
    templates = {tpl.template_id: tpl for tpl in list_promotion_gate_templates()}
    assert set(templates.keys()) == {
        "clinical_conservative",
        "screen_ruleout",
        "confirm_rulein",
        "research_exploratory",
    }

    assert [(g.metric, g.op, g.threshold) for g in templates["clinical_conservative"].gates] == [
        ("Balanced Accuracy", ">=", 0.80),
        ("Sensitivity", ">=", 0.85),
        ("MCC", ">=", 0.60),
    ]
    assert [(g.metric, g.op, g.threshold) for g in templates["screen_ruleout"].gates] == [
        ("Sensitivity", ">=", 0.95),
        ("ROC AUC", ">=", 0.90),
    ]
    assert [(g.metric, g.op, g.threshold) for g in templates["confirm_rulein"].gates] == [
        ("Specificity", ">=", 0.98),
        ("Precision", ">=", 0.90),
        ("MCC", ">=", 0.65),
    ]
    assert [(g.metric, g.op, g.threshold) for g in templates["research_exploratory"].gates] == [
        ("Balanced Accuracy", ">=", 0.70),
        ("F1 Score", ">=", 0.65),
    ]


def test_manual_gates_override_template_and_record_ignored_template() -> None:
    thresholds = ThresholdsConfig(
        promotion_gate_template="clinical_conservative",
        promotion_gates=[
            PromotionGateSpec(metric="Sensitivity", op=">=", threshold=0.90, scope="both", aggregation="mean")
        ],
    )

    results = evaluate_promotion(
        thresholds,
        technical_metrics={"recall": 0.95},
        technical_per_fold={"recall": [0.92, 0.96]},
        test_metrics={"recall": 0.94},
    )
    template_meta = results["technical_validation"].template

    assert template_meta["template_id"] == "manual_override"
    assert template_meta["ignored_template"]["template_id"] == "clinical_conservative"
    assert template_meta["ignored_template"]["status"] == "ignored_due_to_manual_override"


def test_gate_evaluation_mean_and_min_aggregation() -> None:
    thresholds = ThresholdsConfig(
        promotion_gates=[
            PromotionGateSpec(metric="Sensitivity", op=">=", threshold=0.90, scope="outer", aggregation="mean"),
            PromotionGateSpec(metric="Sensitivity", op=">=", threshold=0.90, scope="outer", aggregation="min"),
        ]
    )

    results = evaluate_promotion(
        thresholds,
        technical_metrics={"recall": 0.92},
        technical_per_fold={"recall": [0.85, 0.95, 0.96]},
        test_metrics={"recall": 0.91},
    )

    per_gate = results["technical_validation"].per_gate_results
    assert per_gate[0].aggregation == "mean"
    assert per_gate[0].passed
    assert per_gate[1].aggregation == "min"
    assert not per_gate[1].passed
    assert not results["technical_validation"].passed


def test_promotion_report_contains_template_layman_and_gate_rows(tmp_path: Path) -> None:
    thresholds = ThresholdsConfig(promotion_gate_template="research_exploratory")
    results = evaluate_promotion(
        thresholds,
        technical_metrics={"balanced_accuracy": 0.75, "f1_macro": 0.70},
        technical_per_fold={"balanced_accuracy": [0.72, 0.78], "f1_macro": [0.68, 0.72]},
        test_metrics={"balanced_accuracy": 0.74, "f1_macro": 0.69},
    )

    report_path = write_promotion_report(
        outdir=tmp_path,
        decision=True,
        reasons=[],
        gate_table=pd.DataFrame([{"phase": "technical_validation", "passed": True, "reasons": ""}]),
        gating_metrics=pd.DataFrame(per_gate_rows(results)),
        report_only_metrics=pd.DataFrame([]),
        promotion_template=results["technical_validation"].template,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "## Promotion Gate" in content
    assert "research_exploratory" in content
    assert "Layman explanation" in content
    assert "balanced_accuracy" in content.lower() or "Balanced Accuracy" in content
