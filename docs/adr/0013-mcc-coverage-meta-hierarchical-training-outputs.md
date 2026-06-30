# ADR 0013: MCC Coverage in Meta and Hierarchical Training Outputs

## Status
Accepted

## Context
Promotion gate templates include `MCC` as a first-class threshold metric (for example,
`confirm_rulein`). Technical validation summaries are aggregated from mode-specific outer
metrics CSV files. In meta mode, `metrics_outer_meta_eval.csv` did not emit an `mcc` column,
which caused `MCC` gate evaluation to resolve as missing/`NaN` despite other decision metrics
being available. Hierarchical mode outer/summary tables also omitted `mcc`.

Relevant module docs:
- `docs/change-management/training/meta.md`
- `docs/change-management/training/hierarchical_cv.md`
- `docs/change-management/projects/orchestrator.md`

## Decision
Additive schema update to training outputs:
- `train_meta_classifier` now writes `mcc` for both train and val rows in
  `metrics_outer_meta_eval.csv`.
- `train_hierarchical` now writes `mcc` for `L1`, `L2_oracle_*`, and `pipeline` rows in
  `metrics_outer_eval.csv`.
- hierarchical `metrics_summary.*` now includes `mcc_mean`/`mcc_std` where values are available.

No existing columns/files are removed or renamed.

## Consequences
### Positive
- `MCC`-based promotion gates evaluate correctly in meta and hierarchical technical validation.
- Decision-metric schema is more consistent across training modes.
- Existing readers that ignore unknown columns remain unaffected.

### Negative / tradeoffs
- Additive CSV schema growth requires downstream code that hardcodes exact column sets to tolerate
  the new `mcc` columns.

## Compatibility
- CLI flags: no changes.
- Artifacts/manifests/bundles:
  - additive columns only in metrics CSVs; no layout or filename changes.
- Migrations/deprecations:
  - none required.

## Testing plan
- Unit/integration-style:
  - `tests/training/test_meta_calibration_policy_integration.py` asserts meta val row includes
    non-null `mcc`.
  - `tests/training/test_hierarchical_no_outer_val.py` asserts hierarchical outer metrics include
    non-null `mcc` for `L1`.
- Existing mode coverage remains:
  - multiclass regression tests already assert `mcc` presence.

## Rollout / release notes
Add `[Unreleased]` changelog note describing additive `mcc` coverage for meta and hierarchical
technical metrics outputs and promotion-gate impact.
