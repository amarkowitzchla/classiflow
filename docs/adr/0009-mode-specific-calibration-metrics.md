# ADR 0009: Mode-Specific Calibration Metrics and Curve Contracts

## Status
Accepted

## Context
Calibration outputs in `metrics/calibration.py` used a single `ece` name for top-1 confidence calibration and a single `brier` key across binary and multiclass behavior. This caused ambiguity in multiclass/meta/hierarchical modes and made calibration interpretation fragile when `y_pred` did not match `argmax(y_proba)`.

Relevant docs:
- `docs/change-management/metrics.md`
- `docs/change-management/inference.md`
- `docs/change-management/training/meta.md`

Constraints:
- Keep reviewer-facing artifacts interpretable across `binary`, `multiclass`, `meta`, and `hierarchical`.
- Preserve one-release backward compatibility for existing metric keys (`ece`, `brier`) and calibration curve path (`calibration_curve.csv`).

## Decision
1. Extend `compute_probability_quality` with explicit `mode` and `binning` arguments and multi-curve outputs:
   - `(metrics: dict, curves: dict[str, DataFrame])`
2. Introduce explicit metric names:
   - top1: `ece_top1`, `mean_confidence_top1`, `accuracy_top1`, `confidence_gap_top1`
   - binary probability calibration: `ece_binary_pos`, `brier_binary`
   - multiclass probability calibration: `ece_ovr_macro`, per-class `ece_ovr__<class>`, `brier_multiclass_sum`, `brier_multiclass_mean`
   - recommended key: `brier_recommended`
3. Add alignment reporting:
   - `pred_alignment_mismatch_rate`
   - `pred_alignment_note`
4. Keep one-release aliases:
   - `ece` -> `ece_top1`
   - `brier` -> `brier_recommended`
   - emit deprecation warnings in code and include deprecation note in project markdown reports.
5. Add quantile binning with duplicate-edge handling for robust calibration curves.
6. Update training/inference integration:
   - pass explicit `mode`
   - persist top1 compatibility curve as `calibration_curve.csv`
   - persist all curves as `calibration_curve_<name>.csv`
   - write metrics under `overall.probability_quality` while preserving legacy flat keys.

## Consequences
### Positive
- Calibration semantics are mode-correct and explicit.
- Meta/hierarchical outputs expose mismatch between final predictions and probability argmax.
- Quantile binning improves stability in small-N and high-confidence runs.

### Negative / tradeoffs
- Metrics payload grows (more keys and curve files).
- Some downstream consumers may need to prefer new keys over compatibility aliases.

## Compatibility
- CLI flags: unchanged.
- Artifacts/manifests/bundles:
  - additive new keys and curve files.
  - `calibration_curve.csv` remains (top1) for compatibility.
- Migrations/deprecations:
  - `ece` and `brier` retained as aliases for one release; move consumers to `ece_top1` and `brier_recommended`.

## Testing plan
- Unit:
  - `tests/metrics/test_calibration.py` covers binary/multiclass/meta/hierarchical behavior and quantile duplicate-edge handling.
- Integration:
  - existing meta/inference calibration alignment suites continue validating class/probability alignment.
- Regression/golden fixtures:
  - alias keys retained to avoid breaking existing artifact readers during transition.

## Rollout / release notes
- Add `CHANGELOG.md` entry under `[Unreleased]` documenting:
  - new calibration keys and curve files
  - compatibility aliases and deprecation notes
  - requirement to migrate threshold/gating configs to explicit keys where needed
