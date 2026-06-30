# ADR 0005: Promotion Gate Templates for Project Recommendations

## Status
Accepted

## Context
`projects` is a high-risk module, and promotion gate semantics are part of the reviewer-facing decision contract. Users needed a productized way to choose clinically meaningful gate bundles without repeatedly hand-editing `registry/thresholds.yaml`.

Prior behavior used legacy required/safety maps and did not provide template provenance in promotion artifacts.

Relevant docs:
- `docs/change-management/projects.md`
- `docs/change-management/cli.md`

## Decision
Introduce first-class promotion gate templates with a single registry source:

- New built-in templates:
  - `clinical_conservative`
  - `screen_ruleout`
  - `confirm_rulein`
  - `research_exploratory`
- Internal fallback template:
  - `default_f1_balacc_v1`
- New gate schema (`promotion_gates`) with per-gate fields:
  - `metric`, `op`, `threshold`, `scope`, `aggregation`, `notes`
- New template selector field:
  - `promotion_gate_template`

Resolution precedence:
1. `promotion_gates` (manual) if present.
2. `promotion_gate_template` if set.
3. Legacy `technical_validation`/`independent_test` required+safety maps.
4. Internal default template (`default_f1_balacc_v1`).

If both manual and template are configured, manual gates win and the template is recorded as `ignored_due_to_manual_override` in provenance metadata.

Evaluation changes:
- Scope-aware checks for outer CV vs independent test vs both.
- Aggregation modes: `mean`, `median`, `min`, `pXX`.
- Structured per-gate outputs are emitted for reports and JSON artifacts.

CLI changes:
- `classiflow project bootstrap --promotion-gate-template <id>`
- `classiflow project recommend --promotion-gate-template <id>`
- `classiflow project bootstrap|recommend --list-promotion-gate-templates`

## Consequences
### Positive
- Standardized, reusable promotion gate presets.
- Stronger provenance: template id/version/source and resolved gate rows included in artifacts.
- Backward-compatible behavior for legacy threshold maps.

### Negative / tradeoffs
- Added schema complexity in `ThresholdsConfig`.
- Report and decision artifacts now include additional gate metadata rows, which downstream parsers must tolerate.

## Compatibility
- CLI flags:
  - Additive new template selection/listing options.
- Artifacts/manifests/bundles:
  - `promotion/decision.yaml` and `promotion/promotion_decision.json` now include template provenance and resolved gate rows.
  - Markdown promotion report includes a dedicated “Promotion Gate” section.
- Migrations/deprecations:
  - No forced migration; legacy required/safety maps still resolve and evaluate.

## Testing plan
- Unit:
  - `tests/unit/test_promotion_gate_templates.py`
  - `tests/unit/test_project_module.py`
- Integration:
  - `tests/integration/test_cli_smoke.py`
- Regression/golden fixtures:
  - Per-gate rows and template provenance are validated in unit tests.

## Rollout / release notes
Add changelog entry describing template IDs, precedence rules, CLI options, and new promotion artifact metadata fields.
