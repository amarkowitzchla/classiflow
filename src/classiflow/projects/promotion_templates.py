"""Built-in promotion gate templates and resolution helpers."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from classiflow.projects.project_models import PromotionGateSpec


class PromotionGateTemplate(BaseModel):
    """Template definition for promotion gates."""

    template_id: str
    display_name: str
    description: str
    layman_explanation: str
    gates: List[PromotionGateSpec] = Field(default_factory=list)
    version: int = 1
    internal: bool = False


TEMPLATE_DEFAULT_F1_BALACC = "default_f1_balacc_v1"


def _template_registry() -> Dict[str, PromotionGateTemplate]:
    return {
        "clinical_conservative": PromotionGateTemplate(
            template_id="clinical_conservative",
            display_name="Conservative Clinical Gate",
            description="Strict gate for broad clinical readiness with balanced performance and robust discrimination.",
            layman_explanation=(
                "The model is consistently right across different groups and doesn't miss too many real cases or overcall too often."
            ),
            gates=[
                PromotionGateSpec(metric="Balanced Accuracy", op=">=", threshold=0.80),
                PromotionGateSpec(metric="Sensitivity", op=">=", threshold=0.85),
                PromotionGateSpec(metric="MCC", op=">=", threshold=0.60),
            ],
            version=1,
        ),
        "screen_ruleout": PromotionGateTemplate(
            template_id="screen_ruleout",
            display_name="Screening / Rule-Out Gate",
            description="Recall-first gate for screening workflows where false negatives are especially costly.",
            layman_explanation="If the model says 'no,' we can trust it - even if it flags extra cases for review.",
            gates=[
                PromotionGateSpec(metric="Sensitivity", op=">=", threshold=0.95),
                PromotionGateSpec(metric="ROC AUC", op=">=", threshold=0.90),
            ],
            version=1,
        ),
        "confirm_rulein": PromotionGateTemplate(
            template_id="confirm_rulein",
            display_name="High-Risk Confirmatory Gate",
            description="Specificity/precision-focused gate for high-risk confirmation where false positives are heavily penalized.",
            layman_explanation="When the model says 'yes,' it's almost always correct.",
            gates=[
                PromotionGateSpec(metric="Specificity", op=">=", threshold=0.98),
                PromotionGateSpec(metric="Precision", op=">=", threshold=0.90),
                PromotionGateSpec(metric="MCC", op=">=", threshold=0.65),
            ],
            version=1,
        ),
        "research_exploratory": PromotionGateTemplate(
            template_id="research_exploratory",
            display_name="Research / Exploratory Gate",
            description="Lenient baseline gate for exploratory model development and research triage.",
            layman_explanation=(
                "The model shows promise and performs better than chance, but isn't ready for clinical use yet."
            ),
            gates=[
                PromotionGateSpec(metric="Balanced Accuracy", op=">=", threshold=0.70),
                PromotionGateSpec(metric="F1 Score", op=">=", threshold=0.65),
            ],
            version=1,
        ),
        TEMPLATE_DEFAULT_F1_BALACC: PromotionGateTemplate(
            template_id=TEMPLATE_DEFAULT_F1_BALACC,
            display_name="Default F1 + Balanced Accuracy",
            description="Internal default template used when no explicit promotion gates are configured.",
            layman_explanation="The model must reach a baseline level on F1 and balanced accuracy.",
            gates=[
                PromotionGateSpec(metric="F1 Score", op=">=", threshold=0.70),
                PromotionGateSpec(metric="Balanced Accuracy", op=">=", threshold=0.70),
            ],
            version=1,
            internal=True,
        ),
    }


def list_promotion_gate_templates(include_internal: bool = False) -> List[PromotionGateTemplate]:
    """List available built-in templates."""
    templates = list(_template_registry().values())
    if not include_internal:
        templates = [tpl for tpl in templates if not tpl.internal]
    return sorted(templates, key=lambda item: item.template_id)


def get_promotion_gate_template(template_id: str) -> PromotionGateTemplate:
    """Fetch a template by id."""
    template = _template_registry().get(template_id)
    if template is None:
        available = ", ".join(sorted(_template_registry().keys()))
        raise ValueError(f"Unknown promotion gate template '{template_id}'. Available: {available}")
    return template
