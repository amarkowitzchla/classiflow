# ADR 0001: Binary Stats Dispatch and Tests

## Status
Accepted

## Context
`stats` needs correct binary (2-class) behavior while preserving multiclass outputs.
This change adjusts test selection logic and introduces per-feature multiple-testing
correction for binary mode. See `docs/change-management/stats.md`.

## Decision
- Add a binary-specific stats pipeline that:
  - runs Shapiro–Wilk per class with `min_n`
  - uses Welch's t-test when both classes pass normality; otherwise Mann–Whitney U
  - applies per-feature p-value adjustment across features using `dunn_adjust`
- Keep multiclass pipeline unchanged (ANOVA/Tukey and Kruskal–Wallis/Dunn).
- Preserve output artifacts/sheet names; binary mode emits single pairwise comparison tables.

## Consequences
### Positive
- Correct binary test selection and p-value adjustment.
- Clear separation between binary and multiclass logic.
### Negative / tradeoffs
- Additional code paths to maintain and test.

## Compatibility
- CLI flags: unchanged.
- Artifacts/manifests/bundles: same filenames; binary mode adds/uses adjusted
  p-values for single pairwise tables.
- Migrations/deprecations: none.

## Testing plan
- Unit: `pytest tests/stats/test_effects.py`, `pytest tests/stats/test_binary.py`
- Regression/golden fixtures: not added (no schema changes; outputs are additive for binary mode).

## Rollout / release notes
Add entry under `CHANGELOG.md` → `[Unreleased]` noting binary stats support.
