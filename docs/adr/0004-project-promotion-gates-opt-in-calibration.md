# ADR 0004: Project Promotion Gates Use Explicit User-Configured Metrics

## Status
Accepted

## Context
`classiflow project recommend` evaluated calibration gates (`brier_calibrated`, `ece_calibrated`) even when those thresholds were not present in `registry/thresholds.yaml`.

This made promotion behavior appear inconsistent to users, because implicit defaults in the schema were treated as active gate requirements.

The affected module is `projects` (high risk by change-management policy), specifically promotion gate semantics and user-facing threshold configuration.

## Decision
- Calibration gates are opt-in:
  - `promotion.calibration.brier_max` is enforced only when explicitly configured (non-null).
  - `promotion.calibration.ece_max` is enforced only when explicitly configured (non-null).
- Metric resolution for gate checks is mode-tolerant for F1 naming:
  - `f1` gate checks resolve against `f1_macro`, then `f1_weighted`, then `f1`.
- `recommend` report tables include calibration gate rows only when the corresponding threshold is configured.
- UI project gate scanner mirrors the same behavior (no implicit fallback thresholds for calibration checks).
- Bootstrap adds a user-intent gate profile selector:
  - `--gate-profile balanced|f1|sensitivity`
  - This sets initial `required` and stability defaults by mode/profile while leaving full manual editing available.

## Consequences
### Positive
- Promotion behavior matches `thresholds.yaml` exactly.
- Users can run recommendation without calibration gates unless they intentionally enable them.
- Gate intent is easier to initialize with profile-based defaults.

### Negative / tradeoffs
- Existing projects that relied on implicit calibration defaults no longer enforce those checks unless thresholds are explicitly set.
- Teams that require calibration gating must ensure those keys are present in project thresholds.

## Compatibility
- CLI flags:
  - Additive: `project bootstrap --gate-profile`.
  - No removal/rename of existing flags.
- Artifacts/manifests/bundles:
  - No file layout changes.
  - `promotion_decision.json`/report rows may omit calibration checks when not configured.
- Migrations/deprecations:
  - No hard migration required.
  - Recommended action: explicitly set `promotion.calibration.brier_max` and/or `promotion.calibration.ece_max` in projects that require calibration as a gate.

## Testing plan
- Unit:
  - `tests/unit/test_project_module.py`
    - no-calibration-config path passes required metrics without implicit calibration failures
    - single configured calibration metric is enforced independently
- Integration:
  - Existing project integration tests continue to cover workflow orchestration.
- Regression/golden fixtures:
  - Regression added at unit level for explicit-vs-implicit calibration gate behavior.

## Rollout / release notes
- Add a changelog entry under `[Unreleased]` explaining:
  - calibration gates are now explicit opt-in thresholds
  - new bootstrap `--gate-profile` option
  - guidance to configure calibration thresholds explicitly where required
