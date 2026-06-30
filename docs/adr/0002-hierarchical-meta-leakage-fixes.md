# ADR 0002: Hierarchical/Meta Leakage Fixes

## Status
Accepted

## Context
Nested CV invariants require that outer validation data never influences training, tuning, or early stopping. Audits identified:
- Hierarchical training used outer-val for early stopping and inner CV ignored patient grouping.
- Meta OOF feature generation fell back to in-sample predictions.
- Explicit `feature_cols` could include label/patient identifiers.

Relevant docs:
- `docs/change-management/training/hierarchical_cv.md`
- `docs/change-management/training/meta.md`
- `docs/change-management/io.md`

## Decision
- Build early-stopping splits strictly from outer-train, group-aware when `patient_col` is set.
- Use group-aware inner CV for hierarchical tuning with strict patient label checks.
- Remove in-sample fallback for OOF meta-features and drop rows with missing OOF scores.
- Reject `feature_cols` that include label/patient columns.
- Label L2 metrics as oracle-gated (`gate=oracle_l1`).

## Consequences
### Positive
- Eliminates confirmed leakage paths and aligns training with clinical validation standards.
- Clarifies oracle-gated L2 metrics for reviewers.

### Negative / tradeoffs
- Metrics may drop compared to historical runs.
- Meta training may drop rows if OOF scores are missing.

## Compatibility
- CLI flags: unchanged.
- Artifacts/manifests/bundles: filenames unchanged; L2 metrics labeling changes output values in `metrics_outer_eval.*`.
- Migrations/deprecations: none; document baseline shifts in release notes.

## Testing plan
- Unit: new tests for early-stopping split isolation, group-aware inner CV, OOF no-fallback behavior, and `feature_cols` guards.
- Integration: existing meta inference alignment tests recommended after change.
- Regression/golden fixtures: not added; consider for future baselines.

## Rollout / release notes
- Add entries to `CHANGELOG.md` under Fixed for leakage corrections and guards.
