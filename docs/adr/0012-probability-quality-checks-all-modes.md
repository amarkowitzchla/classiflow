# ADR 0012: Probability-Quality Checks for Binary/Multiclass/Hierarchical Technical Validation

## Status
Accepted

## Context
ADR 0011 introduced standardized probability-quality checks in technical validation reports, but the implementation was effectively meta-centric. Binary, multiclass, and hierarchical technical runs did not consistently publish fold-level probability-quality payloads in `run.json`, and checks did not apply mode-specific semantics (binary positive-class calibration and hierarchical postprocessing mismatch interpretation).

This change touches high-risk modules (`metrics`, `training`, `projects`) and artifact contracts (`run.json`, calibration curve CSV outputs, technical report content).

## Decision
1. Expand training artifact emission so `binary`, `multiclass`, `meta`, and `hierarchical` technical runs publish fold-level probability-quality payloads under:
   - `run.json :: artifact_registry.probability_quality.final_variant`
   - `run.json :: artifact_registry.probability_quality.folds.*`
2. Standardize curve artifact outputs for all modes:
   - Keep `calibration_curve.csv` as top1 final-variant compatibility path.
   - Add/keep mode-specific additive files such as:
     - `calibration_curve_top1_<variant>.csv`
     - `calibration_curve_binary_pos_<variant>.csv` (binary)
     - `calibration_curve_ovr_*_<variant>.csv` (multiclass/meta/hierarchical)
3. Make rule engine mode-specific:
   - Binary occupancy uses `binary_pos` curve when present.
   - Binary helpfulness/shift checks use `brier_binary`, `log_loss`, `ece_binary_pos`.
   - Multiclass/meta/hierarchical keep top1 + OVR diagnostics with low-power caveat when class support is unavailable.
   - Add `PQ-H001` for hierarchical argmax-vs-final-label mismatch using `pred_alignment_mismatch_rate`.
4. Update project calibration defaults:
   - `calibration.binning: quantile`
   - `calibration.policy.apply_to_modes: [binary, multiclass, hierarchical, meta]`
   - `calibration.policy.probability_quality_checks.enabled: true`
   - `calibration.policy.probability_quality_checks.apply_to_modes: [binary, multiclass, hierarchical, meta]`
5. Ensure technical report rendering remains evidence-traceable and mode-aware.

## Consequences
### Positive
- Probability-quality diagnostics are available and comparable across all task modes.
- Binary and hierarchical reports avoid misleading interpretation by using mode-correct diagnostics.
- Artifact contracts stay additive and backward-compatible (`calibration_curve.csv` retained).

### Negative / tradeoffs
- Additional CSV artifacts are produced per fold/variant.
- Rule output may include more WARN/INFO rows in previously uninstrumented modes.

## Compatibility
- CLI flags: no breaking removals.
- Artifacts/manifests:
  - additive `run.json` and calibration curve files for non-meta modes
  - compatibility path `calibration_curve.csv` preserved
- Migrations/deprecations: none required for existing readers using compatibility keys/files.

## Testing plan
- Unit:
  - `tests/metrics/test_probability_quality_checks.py`
  - `tests/unit/test_reporting_probability_quality.py`
- Regression coverage:
  - binary uses `binary_pos` occupancy and binary metrics in PQ-004
  - multiclass PQ-006 low-power INFO caveat
  - hierarchical `PQ-H001` trigger on mismatch
  - report section rendering with evidence paths for each mode

## Rollout / release notes
- Add `[Unreleased]` changelog entry describing all-mode probability-quality checks, binary/hierarchical mode-specific diagnostics, and additive curve/manifest outputs.
