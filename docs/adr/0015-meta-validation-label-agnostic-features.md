# ADR 0015: Meta Validation Label-Agnostic Feature Scoring

## Status
Accepted

## Context
`training/meta` must preserve nested-CV leakage invariants. A leakage path existed in outer
validation feature construction: task scores were conditionally populated using `y_va` task
membership, which encodes validation-label structure into sparse meta-feature patterns.

Relevant docs:
- `docs/change-management/training.md`
- `docs/change-management/training/meta.md`

## Decision
- Build validation meta-features by scoring all validation rows for every selected binary task.
- Keep OOF generation for training task-member rows, and overwrite those rows with OOF scores.
- Keep strict no-fallback behavior: missing OOF rows remain `NaN` and are dropped from meta training.

## Consequences
### Positive
- Removes label-conditioned validation feature construction.
- Aligns training-time validation feature semantics with inference behavior (`BinaryPredictor` scores all rows).
- Preserves OOF protections against in-sample optimism in meta training.

### Negative / tradeoffs
- Pairwise/composite task scores are now produced for all rows at validation time, including rows
  whose true labels were outside task training membership; this can shift historical metrics.

## Compatibility
- CLI flags: unchanged.
- Artifacts/manifests/bundles: unchanged filenames and schema; value distributions may shift.
- Migrations/deprecations: none required.

## Testing plan
- Unit:
  - `pytest tests/training/test_meta_oof_no_fallback.py`
    - verifies no OOF fallback
    - verifies validation meta-feature scores do not depend on labels
- Integration:
  - `pytest tests/integration/test_meta_inference_consistency.py`
- Regression/golden fixtures:
  - leakage regression covered by unit test on label-agnostic validation features.

## Rollout / release notes
- Add `[Unreleased]` changelog item under `Fixed` for meta validation leakage guard.
