# ADR 0008: Calibration Metrics Binary Probability Fix

## Status
Accepted

## Context
`compute_probability_quality` in `src/classiflow/metrics/calibration.py` is used by meta training and inference reporting.  
For binary classification, the previous implementation used `label_binarize` with two classes and then required column-count equality with `y_proba`. In sklearn, binary `label_binarize` returns shape `(n_samples, 1)`, while probability arrays are `(n_samples, 2)`. This caused an early return with all calibration metrics as `NaN`.

Affected change-management modules:
- `docs/change-management/metrics.md` (High risk)
- downstream consumers in `inference` and `projects` promotion/reporting paths

## Decision
- Keep public metric keys unchanged (`brier`, `log_loss`, `ece`).
- In `compute_probability_quality`:
  - validate probability matrix shape and finiteness;
  - reject non-probability score matrices (values outside `[0,1]`);
  - for binary classes, compute Brier directly using positive-class probability (`classes[1]`);
  - keep multiclass behavior and class-order contract unchanged.

## Consequences
### Positive
- Binary calibration metrics are now computed instead of silently returning `NaN`.
- Inference and promotion flows that rely on binary probability quality get valid metrics when probabilities are available.
- Invalid score matrices are now explicitly rejected in this function, reducing misleading calibration outputs.

### Negative / tradeoffs
- Callers passing decision-function outputs into probability-quality helpers now get `NaN` (explicitly), not pseudo-metrics.
- Historical binary calibration values may change from `NaN` to real numbers, affecting comparisons.

## Compatibility
- CLI flags: no change.
- Artifacts/manifests/bundles: no schema change; metric keys and filenames unchanged.
- Migrations/deprecations: none required.

## Testing plan
- Unit: `tests/metrics/test_calibration.py`
  - binary path produces finite Brier/log-loss/ECE
  - multiclass perfect case unchanged
  - non-probability matrix rejected
- Training calibration sanity: `tests/training/test_meta_calibration.py`
- Integration class-alignment guard: `tests/integration/test_meta_brier_alignment.py`

## Rollout / release notes
- Add `CHANGELOG.md` entry under `[Unreleased]` describing binary calibration metric fix and non-probability rejection behavior.
