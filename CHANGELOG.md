# Changelog

All notable user-visible changes to `classiflow` must be documented here.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning:
- https://keepachangelog.com/en/1.0.0/
- https://semver.org/spec/v2.0.0.html

Guidelines:
- Record changes that affect CLI behavior, defaults, outputs/artifacts, bundle layouts, manifests, schemas, and APIs.
- Prefer linking changes to PRs/issues and note any migration/compatibility guidance.
- Do not invent historical entries; add entries starting from the point this file is adopted.

## [Unreleased]

### Added

- **Experiment Tracking Integration**: Optional MLflow and Weights & Biases support for all training commands
  - New `tracking` module with pluggable tracker architecture
  - CLI flags: `--tracker`, `--experiment-name`, `--run-name` for all `train-*` commands
  - Automatic logging of parameters, metrics, and artifacts
  - Project workflow integration via `project.yaml` config
  - Optional dependencies: `pip install classiflow[mlflow]` or `pip install classiflow[wandb]`
  - See `docs/TRACKING_GUIDE.md` for full documentation
- **Stats binary mode**: 2-class datasets now dispatch to Welch's t-test or Mann–Whitney U with per-feature p-value adjustment.

### Changed
- Binary inference now emits `predicted_label` and `predicted_proba_*` columns when labels
  are present and scores are probabilities, enabling plots for binary `project run-test`.
- `project recommend` now enforces calibration gates only when explicitly configured in
  `registry/thresholds.yaml` (`promotion.calibration.brier_max` / `ece_max`), removing
  implicit calibration failures when those keys are absent.
- `project recommend` now resolves `f1` thresholds against available metrics
  (`f1_macro` then `f1_weighted` then `f1`) to avoid mode-specific naming failures.
- `project bootstrap` now supports `--gate-profile balanced|f1|sensitivity` to initialize
  threshold defaults by user objective while keeping full manual control in thresholds YAML.
- Added promotion gate templates with provenance-aware evaluation:
  - Built-ins: `clinical_conservative`, `screen_ruleout`, `confirm_rulein`, `research_exploratory`
  - New thresholds schema fields: `promotion_gate_template`, `promotion_gates`
  - Precedence: manual `promotion_gates` overrides template and records
    `ignored_due_to_manual_override`
  - New CLI options: `--promotion-gate-template`, `--list-promotion-gate-templates`
    on `project bootstrap` and `project recommend`
  - Promotion artifacts now include template metadata, layman explanation, resolved gate rows,
    and per-gate pass/fail details for technical validation and independent test.

### Deprecated

### Removed

### Fixed
- Prevented hierarchical early stopping from using outer validation data; inner CV now respects patient groups.
- Removed in-sample fallback for meta OOF features and drop missing OOF rows during meta training.
- Guarded against label/patient leakage via explicit `feature_cols`.

### Security
