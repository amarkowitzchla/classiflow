# ADR 0003: Binary inference predicted_label and predicted_proba columns

## Status
Accepted

## Context
Binary inference outputs only `{task}_score` and `{task}_pred`, while multiclass/meta runs
emit `predicted_label` and `predicted_proba_{class}`. This mismatch prevents plotting and
downstream consumers from treating binary outputs consistently. This change touches the
inference output schema, which is high risk. See `docs/change-management/inference.md`
and `docs/change-management/inference/run_inference.md`.

## Decision
Augment binary inference outputs with:
- `predicted_label` derived from `{task}_pred` and label-derived class names.
- `predicted_proba_{class}` and `predicted_proba` when `{task}_score` is a probability
  (scores in `[0, 1]`).

The augmentation only runs when labels are present in the input data. If labels are not
available or scores are not probabilities, the new probability columns are omitted.

## Consequences
### Positive
- Binary runs emit plot-compatible columns, enabling ROC/confusion/score plots.
- Downstream consumers can rely on consistent prediction columns across run types.

### Negative / tradeoffs
- Adds conditional column emission based on label availability and score range.
- Downstream code must tolerate new columns in binary outputs.

## Compatibility
- CLI flags: no change.
- Artifacts/manifests/bundles: `predictions.csv` gains additional columns for binary runs.
- Migrations/deprecations: none required; change is additive.

## Testing plan
- Unit: `pytest tests/inference/test_binary_predictions.py`
- Integration: `pytest tests/integration/test_meta_inference_consistency.py`
- Regression/golden fixtures: none added (add if future regressions occur).

## Rollout / release notes
Add a `[Unreleased]` CHANGELOG entry noting binary inference now emits
`predicted_label`/`predicted_proba_*` when labels are present and scores are probabilities.
