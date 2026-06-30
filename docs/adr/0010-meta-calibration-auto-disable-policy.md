# ADR 0010: Meta Calibration Auto-Disable Policy and Decision Artifacts

## Status
Accepted

## Context
ADR 0009 introduced mode-specific calibration metrics and multi-curve outputs, but meta training still treated calibration as mostly opt-in/always-on and did not encode explicit rollback reasons in artifacts.

This made high-accuracy underconfident folds and small-sample folds hard to review, and could retain calibrated probabilities even when Brier/log-loss/ECE guardrails regressed.

## Decision
1. Add policy-driven calibration enablement for meta training via `calibration_enabled`:
   - `false`: skip calibration.
   - `true`: attempt calibration, then apply R4 guardrail unless `force_keep`.
   - `auto`: apply R1-R4 for configured modes (`meta` by default).
2. Implement policy rules in `metrics/calibration_policy.py` and evaluate uncalibrated/calibrated metrics on the same validation split.
3. Roll back final probabilities/predictions to uncalibrated outputs when policy disables calibration.
4. Persist explicit decision artifacts per fold:
   - `calibration_summary.json` now includes:
     - `overall.probability_quality.uncalibrated`
     - `overall.probability_quality.calibrated`
     - `overall.probability_quality.final_variant`
     - `overall.probability_quality.calibration_decision`
5. Persist compatibility + diagnostic curves:
   - `calibration_curve.csv` remains mapped to final variant top1 curve.
   - Add variant-tagged curves: `calibration_curve_<name>_uncalibrated.csv`,
     `calibration_curve_<name>_calibrated.csv`,
     plus explicit top1 aliases.
6. Add run-level summary in `run.json` under `artifact_registry.probability_quality`.

## Consequences
### Positive
- Calibration behavior is explicit, reviewable, and reversible.
- Probability-quality artifacts always show both variants and final selection.
- Compatibility path `calibration_curve.csv` remains stable.

### Tradeoffs
- Additional artifact files and decision payload increase output volume.
- `ProjectConfig.calibration` migrated from legacy `calibrate_meta` boolean to `enabled` enum; legacy key is normalized.

## Compatibility
- Backward compatibility:
  - legacy `calibration.calibrate_meta` still accepted via normalization.
  - `calibration_curve.csv` preserved.
- Additive artifacts:
  - variant-specific curve files
  - decision payloads in fold summaries and manifest artifact registry

## Testing
- Unit: `tests/metrics/test_calibration_policy.py`
- Training integration: `tests/training/test_meta_calibration_policy_integration.py`
- Existing calibration training tests updated for the new calibration helper API.

## Rollout Notes
- Prefer `calibration.enabled: auto` for meta runs.
- Consumers should read `overall.probability_quality.final_variant` and `calibration_decision` instead of inferring from `calibration_enabled` alone.
