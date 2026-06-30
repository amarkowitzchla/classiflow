# ADR 0006: Ensure Multiclass Technical Metrics Include Promotion Gate Fields

## Status
Accepted

## Context
Promotion gate templates can require `Sensitivity` and `MCC`.

For multiclass technical validation runs (including torch estimators via `estimator_mode=torch_only`),
`metrics_outer_multiclass_eval.csv` previously omitted decision-focused fields (`sensitivity`, `mcc`, etc.).
This caused `project recommend` to report missing gate metrics even when model predictions were valid.

Affected high-risk modules:
- `training` (multiclass metric export semantics)
- `projects` (technical metrics summarization consumed by promotion gates)

## Decision
1. Extend multiclass outer-fold metric computation to always include:
- `sensitivity`, `specificity`, `ppv`, `npv`, `recall`, `precision`, `mcc`

2. Keep existing multiclass metrics unchanged:
- `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `roc_auc_ovr_macro`

3. Extend project technical metric summarization to include `mcc` in additive extraction from outer CSVs.

## Consequences
### Positive
- Promotion templates requiring `Sensitivity` and `MCC` evaluate correctly on multiclass technical runs.
- Torch and sklearn multiclass estimators share consistent promotion-relevant output schema.

### Negative / tradeoffs
- Additional columns in multiclass outer metrics CSVs (additive schema growth).

## Compatibility
- Artifacts/manifests/bundles:
  - Additive columns in `metrics_outer_multiclass_eval.csv`; no removals/renames.
- CLI flags:
  - No changes.
- Migration/deprecations:
  - None required; rerun technical validation to populate new columns in historical projects.

## Testing plan
- `pytest tests/training/test_multiclass.py`
- `pytest tests/training/test_multiclass_hardening.py`
- `pytest tests/unit/test_project_module.py`

## Rollout / release notes
Document that multiclass technical runs now emit decision-facing metrics needed by promotion gate templates and that existing run folders require re-running technical validation to backfill these fields.
