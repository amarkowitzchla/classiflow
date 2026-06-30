# ADR 0011: Probability-Quality Checks in Technical Validation Reports

## Status
Accepted

## Context
`technical_validation_report.md` previously listed calibration metrics but did not provide standardized, evidence-traceable interpretation for ECE/Brier behavior. Reviewers needed consistent guidance for common calibration failure modes (low occupancy, over/underconfidence, calibration regressions) while preserving ADR 0009 contracts and compatibility aliases.

Relevant docs:
- `docs/change-management/metrics.md`
- `docs/change-management/projects.md`
- `docs/change-management/projects/orchestrator.md`
- `docs/adr/0009-mode-specific-calibration-metrics.md`

Constraints:
- Keep artifact contracts stable (`run.json`, `calibration_curve.csv`, variant curve files).
- Use additive reporting changes.
- Cite exact artifacts/fields used by each diagnostic.

## Decision
1. Add `src/classiflow/metrics/probability_quality_checks.py` with:
   - `ProbQualityRuleResult` dataclass.
   - Safe loaders for `run.json` probability-quality payload and curve CSVs.
   - Standardized rules `PQ-001` through `PQ-007` with severities, measured values, thresholds, evidence, and actionable recommendations.
2. Integrate checks into `write_technical_report`:
   - New section: `Probability Quality Checks (ECE/Brier)`.
   - Summary table plus per-rule details (condition, measured, thresholds, evidence, recommendations).
   - Add “Promotion gate impact” note and warning when promotion gates reference `ece*`/`brier*` metrics.
3. Add config-driven threshold overrides sourced from:
   - `calibration.policy.probability_quality_checks`
   - `calibration.policy.thresholds.probability_quality_checks`
4. Keep behavior fail-safe:
   - Missing artifacts produce no emitted warnings (rules skip).
   - Independent-test shift rule (`PQ-005`) only evaluates when independent metrics are provided.

## Consequences
### Positive
- Reviewers get consistent, rule-based calibration diagnostics with traceable artifact evidence.
- Technical reports now provide explicit next-step guidance for common probability-quality patterns.
- Thresholds are tunable without changing code.

### Negative / tradeoffs
- Report verbosity increases.
- New logic depends on availability and consistency of fold-level calibration artifacts.

## Compatibility
- CLI flags: unchanged.
- Artifacts/manifests/bundles:
  - No schema renames/removals.
  - Report content is additive.
  - Existing `calibration_curve.csv` and alias keys remain supported.
- Migrations/deprecations:
  - None required.

## Testing plan
- Unit:
  - `tests/metrics/test_probability_quality_checks.py`
  - `tests/unit/test_reporting_probability_quality.py`
- Integration:
  - `tests/integration/test_technical_validation_report_probability_quality.py`
- Regression/golden fixtures:
  - Synthetic run fixtures validate triggered rule IDs and evidence rendering.

## Rollout / release notes
- Add `CHANGELOG.md` entry under `[Unreleased]` for technical report probability-quality checks, evidence citations, threshold overrides, and promotion-gate warning behavior.
