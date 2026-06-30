# ADR 0014: Meta Under-the-Hood Binary Learner Health Diagnostics

## Status
Accepted

## Context
Meta-mode technical validation reports previously focused on final meta-classifier outputs. Reviewers lacked standardized diagnostics for underlying one-vs-rest binary learners, creating risk of silent collapse (degenerate probabilities, unstable folds, leakage-like perfect AUC patterns) that could still appear masked in aggregate meta metrics.

This change touches high-risk modules (`training`, `metrics`, `projects`) and extends artifact contracts (`run.json`, technical report content, new CSV/JSON/PNG outputs).

## Decision
1. Persist fold-level OVR base learner probability outputs in meta training:
   - `fold{N}/binary_{variant}/base_ovr_proba_fold{N}.npz`
2. Add binary learner diagnostics artifacts at technical-run root:
   - `binary_learners_manifest.json`
   - `binary_learners_metrics_by_fold.csv`
   - `binary_learners_metrics_summary.csv`
   - `binary_learners_warnings.json`
   - `ovo_auc_matrix.csv`
   - `plots/binary_ovr_roc_<class>.png`
   - `plots/binary_ovr_roc_all_classes.png`
   - `plots/ovo_auc_matrix.png`
3. Implement rule-based warnings `BL-001`..`BL-006` with severity (`INFO`/`WARN`/`ERROR`) and evidence paths:
   - collapse / degenerate decision rate / no-signal AUC / high fold variance / suspiciously perfect / low-power class
4. Add report section:
   - `Binary Learner Health Report (Meta Under-the-Hood)` in `technical_validation_report.md`
   - includes summary table, warning table, artifact links, and impact paragraph
5. Extend `run.json` with additive pointer block:
   - `artifact_registry.binary_learners` (paths to diagnostics artifacts)

## Consequences
### Positive
- Reviewers can inspect base learner reliability and instability directly.
- Warnings are standardized, evidence-cited, and actionable.
- OVR and OVO plots improve interpretability of meta behavior.

### Negative / tradeoffs
- Additional per-run artifacts and plots increase storage footprint.
- Technical report length increases for meta runs.

## Compatibility
- CLI flags: no breaking changes.
- Artifacts/manifests:
  - additive files only; existing files retained.
  - `run.json` gains additive `artifact_registry.binary_learners`.
- Migrations/deprecations: none required.

## Testing plan
- Unit:
  - `tests/metrics/test_binary_learners.py`
- Integration/regression:
  - `tests/integration/test_technical_validation_report_probability_quality.py`
  - `tests/training/test_meta_calibration_policy_integration.py`

## Rollout / release notes
- Add `[Unreleased]` changelog entry describing new meta binary-learner diagnostics artifacts, warnings, and technical report section.
