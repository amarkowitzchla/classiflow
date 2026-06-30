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
- **`AGENTS.md`, `docs/adr/`, `docs/change-management/`** are now properly tracked in git. These
  critical operational files were never committed due to an over-broad `docs/*` gitignore rule.
  No behavioral change; purely a repository hygiene fix.

### Changed
- **`.gitignore` rewrite**: removed erroneous `docs/*`, `AGENTS.md`, and duplicate `derived/`
  entries; added `classiflow-ui/node_modules/`, `.classiflow/ui.db`, `*.bak`, and explicit
  `*.parquet` rules; fixed `.classflow/ui.db` typo. This prevents unintentional future omissions
  of documentation and policy files.
- **`pyproject.toml` tooling fixes** (no behavior change):
  - Ruff `select`/`ignore` moved to `[tool.ruff.lint]` section (deprecated top-level syntax).
  - `N803`/`N806` added to ruff ignore list (scikit-learn/ML convention for uppercase `X`, `y`
    argument and variable names).
  - `__init__.py` files exempt from `F401` (intentional public re-exports).
  - `black` `target-version` set to `["py39", "py310", "py311"]`; removed `py312` which the
    installed black 22.x does not recognise.
  - `Documentation` URL added to `[project.urls]`.
- **Code formatting**: `black` and `ruff --fix` applied across `src/` and `tests/`; 107 files
  reformatted and 260 import/f-string issues auto-fixed. No logic changed.
- **`plots/schemas.py`**: Pydantic `class Config` inner classes converted to `model_config =
  ConfigDict(...)` to silence Pydantic V2 deprecation warnings.

### Fixed
- **`tasks/builder.py`**: Missing `Literal` import for `add_composite_task` type annotation
  (caused `NameError` when `get_type_hints()` was called on the method).
- **`ui_api/scanner.py`**: Missing `Any` import used in `_numeric_metrics` annotation.
- **`cli/backfill.py`**: Removed dead variables `fold_data`, `fold_num`, `metrics_file`, and
  `proba_file` — these were assigned inside the fold loop but never consumed (incomplete code
  paths with no behavioral effect).
- **`cli/infer.py`**: Removed unused `bundle_loader = None` sentinel; the variable was never
  read after assignment.
- **`cli/main.py`**: Dropped unused return-value assignment from `train_binary_task(config)`;
  the CLI branch only needs the side-effects.
- **Repository**: Removed `.classiflow/ui.db` (SQLite runtime database) and
  `tests/stats/test_normality.py.bak` (backup file) from git tracking; added proper gitignore
  rules so they stay untracked.
- **`classiflow-ui/node_modules/`**: Removed 15 000+ vendored JS dependency files from git
  tracking; added `classiflow-ui/node_modules/` to `.gitignore`.

### Known pre-existing failures
- `tests/integration/test_meta_class_alignment_deep.py::test_full_pipeline_class_alignment`:
  Brier score mismatch (`0.023 ≥ 1e-6`). This failure predates this release and is in a
  High-risk module (inference/metrics). Tracked for future investigation; no changes were made
  to the affected code paths.

### Added

- **Bagged final estimator strategy for binary and direct multiclass workflows**:
  - New CLI flags on `train-binary` and `train-multiclass`:
    `--final-estimator-strategy` and `--bagging-*`.
  - New project config fields:
    `models.final_estimator_strategy` and `models.bagging_*`.
  - Bagging is intentionally not supported for `task.mode=meta` or `hierarchical`.
- **Expanded torch MLP tuning grid**:
  - New `--expanded-mlp-tuning-grid` flag on `train-binary`, `train-meta`,
    and `train-multiclass`.
  - New project config field: `models.expanded_mlp_tuning_grid`.
  - Expanded grids now include CCIX-style MLP hyperparameter axes and nearby values,
    including `activation`, `use_batchnorm`, `hidden_dim`, `n_layers`, `dropout`,
    `lr`, `weight_decay`, `batch_size`, and `epochs`.
- **Bagging diagnostics and final-bundle visibility in inference/UI**:
  - Independent-test inference now exports `bagging_summary.json` and
    `metrics/bag_member_metrics.csv` when bagged estimators are used.
  - UI API payloads now include bag-member metrics and selected final-bundle model
    configuration summaries (strategy/execution/selected model entries).
  - The React run page now includes a dedicated bagging tab and richer
    hierarchical level metric sections (`L1`, `L2`, `L2_by_branch`, `pipeline`).
- **Experiment Tracking Integration**: Optional MLflow and Weights & Biases support for all training commands
  - New `tracking` module with pluggable tracker architecture
  - CLI flags: `--tracker`, `--experiment-name`, `--run-name` for all `train-*` commands
  - Automatic logging of parameters, metrics, and artifacts
  - Project workflow integration via `project.yaml` config
  - Optional dependencies: `pip install classiflow[mlflow]` or `pip install classiflow[wandb]`
  - See `docs/TRACKING_GUIDE.md` for full documentation
- **Stats binary mode**: 2-class datasets now dispatch to Welch's t-test or Mann–Whitney U with per-feature p-value adjustment.
- **Project config UX commands**:
  - `classiflow config show --mode ... --engine ...`
  - `classiflow config explain path.to.field`
  - `classiflow config validate project.yaml`
  - `classiflow config normalize project.yaml --out ...`

### Changed
- PyTorch dependency floor is now `>=2.7.0` to align with official NVIDIA Blackwell
  support, and install docs now call out the required CUDA 12.8 (`cu128`) wheels for
  Blackwell GPU environments.
- `project bootstrap` now accepts `--expanded-mlp-tuning-grid`, final-estimator bagging
  flags, and `--technical-final-estimator-strategy`, allowing project YAML generation
  without manual edits for expanded MLP tuning or bagged final models.
- `project run-technical` now uses `models.technical_final_estimator_strategy`
  (default `single`) instead of always mirroring the final bundle estimator strategy.
- Torch MLP tuning is now unified across the registry-backed binary/meta paths and the
  direct multiclass path, reducing drift between torch training modes.
- Torch MLP implementations now support tunable `activation` and `use_batchnorm`
  hyperparameters so the expanded grid can mirror the CCIX classifier MLP search space.
- `calibration.method` now accepts `temperature` for torch learners across
  `train-*` and `project` workflows.
- The Streamlit training UI now exposes torch backend controls, expanded MLP tuning,
  and binary bagging options; the UI API now surfaces project model settings in
  project dashboard responses.
- Project YAML runtime settings now use `execution.*` (`engine`, `device`, `torch`) instead of top-level
  `backend/device/torch_*` keys for new configs.
- `project bootstrap` now supports explicit runtime selection with `--engine`, `--device`, and
  `--show-options`, and emits mode/engine-aware minimal YAML.
- `project bootstrap` now supports meta custom task wiring with `--tasks-json` and `--tasks-only`:
  - persisted in `project.yaml` as `task.tasks_json` and `task.tasks_only`
  - validated as meta-only options (`--tasks-only` requires `--tasks-json`)
  - forwarded by `project run-technical` into meta training (`MetaConfig.tasks_json/tasks_only`)
- Multiclass project runtime selection is explicit via `multiclass.backend`
  (`sklearn_cpu`, `torch_*`, `hybrid_sklearn_meta_torch_base`).
- Legacy project configs are normalized to the new schema with warnings; use
  `classiflow config normalize` to persist normalized YAML.
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
- Multiclass technical validation (including torch estimators) now exports promotion-critical
  decision metrics in `metrics_outer_multiclass_eval.csv`:
  `sensitivity`, `specificity`, `ppv`, `npv`, `recall`, `precision`, and `mcc`.
  This prevents template gate failures caused by missing `Sensitivity`/`MCC` in technical summaries.
- Calibration metrics are now mode-specific across `binary`, `multiclass`, `meta`, and `hierarchical`:
  - Added explicit keys: `ece_top1`, `ece_binary_pos`, `ece_ovr_macro`,
    `brier_binary`, `brier_multiclass_sum`, `brier_multiclass_mean`, `brier_recommended`,
    `pred_alignment_mismatch_rate`, and `pred_alignment_note`.
  - Added multi-curve calibration exports (`calibration_curve_<name>.csv`) while preserving
    legacy `calibration_curve.csv` (top1) for compatibility.
  - `overall` inference metrics now include `probability_quality` namespace with detailed
    calibration metrics.
  - Backward-compatible aliases retained for one release: `ece` -> `ece_top1`,
    `brier` -> `brier_recommended` (with deprecation warnings).
- Meta calibration now supports policy-driven auto-disable with explicit artifact reporting:
  - New `calibration.enabled` (`auto|true|false`) and policy thresholds (`R1`-`R4`) for
    calibration rollback decisions.
  - Meta fold artifacts now persist both uncalibrated/calibrated probability-quality metrics,
    final variant selection, and decision reasons in `calibration_summary.json`.
  - Compatibility `calibration_curve.csv` is preserved and now tracks the final selected variant,
    with additional variant-tagged diagnostic curves.
  - `run.json` now includes fold-level probability-quality decision summaries under
    `artifact_registry.probability_quality`.
- Technical validation reports now include a standardized
  `Probability Quality Checks (ECE/Brier)` section:
  - Rule IDs `PQ-001` to `PQ-007` evaluate occupancy, over/underconfidence, calibration regressions,
    optional test-shift checks, weak OVR class calibration, and near-perfect/high-ECE reviewer guards.
  - Each triggered rule includes severity, measured values, thresholds, artifact evidence pointers
    (`run.json` field paths and calibration curve CSV paths), and actionable recommendations.
  - Added configurable threshold overrides via
    `calibration.policy.probability_quality_checks` or
    `calibration.policy.thresholds.probability_quality_checks`.
  - Report now warns when promotion gates reference `ece*`/`brier*` metrics and reminds users to
    use explicit key names (`ece_top1` vs `ece_ovr_macro`) with adequate sample support.
- Probability-quality checks and calibration diagnostics now run across all task modes
  (`binary`, `multiclass`, `hierarchical`, `meta`) in technical validation:
  - `run.json` now carries fold-level probability-quality payloads for non-meta modes too.
  - Added additive variant-tagged curve artifacts for top1/binary-pos/OVR outputs while preserving
    `calibration_curve.csv` compatibility.
  - Binary checks are positive-class calibrated (`ece_binary_pos`, `brier_binary`) and occupancy now
    prefers `binary_pos` reliability curves.
  - Hierarchical mode adds `PQ-H001` to interpret argmax probability vs postprocessed final-label mismatch.
- Meta and hierarchical technical validation outputs now include `mcc` in outer metrics CSVs:
  - `metrics_outer_meta_eval.csv` now emits `mcc` for train/val rows.
  - `metrics_outer_eval.csv` (hierarchical) now emits `mcc` for `L1`, `L2_oracle_*`, and `pipeline` rows,
    and hierarchical `metrics_summary.*` includes additive `mcc_mean`/`mcc_std`.
  - This fixes `MCC` promotion-gate evaluations resolving to missing/`NaN` in confirmatory workflows.
- Meta technical validation now publishes under-the-hood binary learner diagnostics:
  - New artifacts: `binary_learners_manifest.json`, `binary_learners_metrics_by_fold.csv`,
    `binary_learners_metrics_summary.csv`, `binary_learners_warnings.json`, `ovo_auc_matrix.csv`,
    `plots/binary_ovr_roc_<class>.png`, `plots/binary_ovr_roc_all_classes.png`,
    and `plots/ovo_auc_matrix.png`.
  - Per-fold OVR base learner probabilities are now persisted at
    `fold{N}/binary_{variant}/base_ovr_proba_fold{N}.npz`.
  - `run.json` now includes additive pointers under `artifact_registry.binary_learners`.
  - `technical_validation_report.md` now includes
    `Binary Learner Health Report (Meta Under-the-Hood)` with class-level summary, BL-001..BL-006
    warnings, evidence paths, and interpretation impact notes.

### Deprecated

### Removed

### Fixed
- Prevented hierarchical early stopping from using outer validation data; inner CV now respects patient groups.
- Removed in-sample fallback for meta OOF features and drop missing OOF rows during meta training.
- Removed validation-label-conditioned meta feature gating in `train-meta`:
  validation task scores are now computed for all rows, and outer-validation feature construction
  no longer uses `y_va` to decide score population.
- Guarded against label/patient leakage via explicit `feature_cols`.
- Fixed multiclass independent-test crashes when the test set omits one or more
  model classes: metrics and probability plots now skip unsupported class
  combinations with explicit warnings.
- Fixed hierarchical report/gate metric resolution by emitting level-specific
  payloads (`L1`, `L2`, `L2_by_branch`, `pipeline`) for technical validation and
  independent-test summaries.
- Fixed hierarchical ROC/PR artifact generation failures on sparse/degenerate
  folds (including sklearn `IndexError` edge cases) with safe fallbacks and
  explicit "unavailable" annotations when curves cannot be computed.
- Fixed binary label post-processing so non-binary numeric predictions are treated
  as missing labels instead of being coerced into positive/negative classes.
- Fixed binary probability calibration quality computation:
  - `compute_probability_quality` now correctly computes binary Brier score/log-loss/ECE instead of returning `NaN` due to binary `label_binarize` shape mismatch.
  - probability-quality helpers now reject non-probability score matrices (outside `[0,1]`) to avoid misleading calibration metrics.

### Security
