# projects

## Objective
- Provide an end-to-end **project workflow layer** for clinical test development:
  - dataset registration
  - technical validation runs (nested CV training + artifacts)
  - feasibility stats/viz
  - independent test inference
  - promotion gate evaluation and final model/bundle creation
- Define a stable on-disk project layout consumed by the UI API and downstream automation.

## Public Interfaces
- Config and schemas (`src/classiflow/projects/project_models.py`):
  - `ProjectConfig.load(path: Path) -> ProjectConfig`
  - `ProjectConfig.load_with_warnings(path: Path) -> (ProjectConfig, list[str])`
  - `ProjectConfig.save(path: Path) -> None`
  - `ProjectConfig.scaffold(...) -> ProjectConfig`
  - `normalize_project_payload(payload: dict) -> (normalized: dict, warnings: list[str])`
  - `ThresholdsConfig` (threshold schema for promotion gates)
  - `DatasetRegistry` (registry schema for datasets)
  - `StabilityGate` (CV stability thresholds)
- Filesystem helpers (`src/classiflow/projects/project_fs.py`):
  - `ProjectPaths(root: Path)` and `ProjectPaths.ensure()`
  - `project_root(base_dir: Path, project_id: str, name: str) -> Path`
  - `choose_project_id(name: str, override: str|None = None) -> str`
- Dataset registry (`src/classiflow/projects/dataset_registry.py`):
  - `register_dataset(registry_path: Path, config: ProjectConfig, dataset_type: str, manifest_path: Path, git_hash: str|None = None) -> DatasetEntry`
  - `verify_manifest_hash(manifest_path: Path, expected_hash: str) -> bool`
- Orchestrator (exported from `src/classiflow/projects/__init__.py`, implemented in `src/classiflow/projects/orchestrator.py`):
  - `run_technical_validation(paths: ProjectPaths, config: ProjectConfig, run_id: str|None = None, compare_smote: bool = False) -> Path`
  - `run_feasibility(paths: ProjectPaths, config: ProjectConfig, run_id: str|None = None, ...) -> Path`
  - `build_final_model(paths: ProjectPaths, config: ProjectConfig, ...) -> Path`
  - `run_independent_test(paths: ProjectPaths, config: ProjectConfig, ...) -> Path`
- Promotion gates (`src/classiflow/projects/promotion.py`):
  - `evaluate_promotion(thresholds: ThresholdsConfig, technical_metrics: dict, technical_per_fold: dict, test_metrics: dict) -> dict[str, GateResult]`
  - `promotion_decision(results: dict[str, GateResult]) -> (passed: bool, reasons: list[str])`
  - `normalize_metric_name(name: str) -> str`
  - `resolve_promotion_gates(thresholds: ThresholdsConfig) -> dict` (manual/template/default resolution)
- Promotion gate templates (`src/classiflow/projects/promotion_templates.py`):
  - `list_promotion_gate_templates(include_internal: bool = False) -> list[PromotionGateTemplate]`
  - `get_promotion_gate_template(template_id: str) -> PromotionGateTemplate`
- CLI:
  - `classiflow project ...` in `src/classiflow/cli/project.py`

Sub-docs:
- `docs/change-management/projects/orchestrator.md`

## Inputs
- Project directory structure (created by CLI `project init` / `project bootstrap`):
  - `<projects_root>/<PROJECT_ID>__<slug>/project.yaml`
  - `<project>/registry/datasets.yaml`, `thresholds.yaml`, `labels.yaml`, `features.yaml`
  - `<project>/runs/<phase>/<run_id>/...`
- Dataset manifests (CSV) registered into the project:
  - must contain at least the label column (`config.key_columns.label`)
  - may contain patient/sample IDs, etc.
- Thresholds config:
  - defines required and safety thresholds and calibration constraints.
  - calibration gates are opt-in; only explicitly configured calibration thresholds are enforced.
  - supports template-driven gating via:
    - `promotion_gate_template: <template_id>`
    - `promotion_gates: [ {metric, op, threshold, scope, aggregation, notes?}, ... ]`
- Project config runtime block:
  - `execution.engine`: `sklearn|torch|hybrid`
  - `execution.device` + `execution.torch.*` required when engine is `torch|hybrid`
  - legacy keys (`backend`, `device`, `torch_*`, `model_set`) are normalized with warnings
- Meta task customization in `task` block:
  - `task.tasks_json`: optional path to JSON task spec (composite/custom task definitions)
  - `task.tasks_only`: when `true`, use only tasks from `task.tasks_json` and skip auto OvR/pairwise
  - valid only for `task.mode=meta`
- Project calibration block:
  - `calibration.enabled`: `auto|true|false` (legacy `calibrate_meta` is normalized)
  - `calibration.policy`: apply modes, optional force-keep override, R1-R4 thresholds, and
    `probability_quality_checks` controls (`enabled`, `apply_to_modes`, optional thresholds override)

## Outputs
- Registry updates:
  - `registry/datasets.yaml` (hashes, schema summaries, stats)
  - `registry/thresholds.yaml` (promotion gates)
- Run directories:
  - technical validation runs with training artifacts and lineage
  - feasibility runs with stats outputs
  - independent test runs with inference outputs
  - final model runs with selected configs, sanity checks, and bundles
- Promotion artifacts:
  - decision summaries and reports (see orchestrator/reporting modules)

## Internal Workflow
- High level:
  - register datasets (hash + schema inference)
  - run technical validation (train models, compute fold metrics, optional SMOTE comparison)
  - evaluate promotion gates using normalized metric names and stability/calibration checks
  - train final model from scratch using selected configs (`projects/final_train.py`)
  - run independent test via inference pipeline to obtain test metrics
- On-disk layout is intentionally structured so:
  - UI scanner can discover projects, phases, and artifacts
  - bundles can be created from run directories

## Dependencies
- Upstream callers:
  - CLI `classiflow project ...`
  - UI API reads projects/runs/artifacts for browsing (`ui_api/scanner.py`)
- Downstream calls:
  - training (`classiflow.training.*`, `training/hierarchical_cv.py`)
  - inference (`classiflow.inference.run_inference`)
  - stats (`classiflow.stats.*`) for feasibility
  - bundles (`classiflow.bundles.create_bundle`)
  - evaluation (`classiflow.evaluation.SMOTEComparison`) for SMOTE compare
- External dependencies: `pandas`, `numpy`, `joblib`, YAML (via `yaml_utils`), optional torch depending on backend.

## Invariants & Safety Constraints
- Project filesystem layout is public API for:
  - `ui_api` scanning/parsing
  - automation scripts and reviewers
- Dataset hashes are used to detect drift/“dirty” registry entries; avoid changing hashing behavior silently.
- Promotion gate semantics must remain stable:
  - metric aliasing and threshold comparisons are part of the review protocol.
- no implicit gate checks beyond the documented fallback:
  - if `promotion_gates` is present, those gates are authoritative.
  - if only `promotion_gate_template` is present, template gates are authoritative.
  - if neither is present and no legacy required/safety thresholds are set, the internal
    `default_f1_balacc_v1` template is applied for consistent reporting provenance.
- Final model training principle is explicit:
  - “train from scratch” for production; do not reuse fold pipelines unless explicitly designed and documented.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new phase or new registry field (additive) | Medium | Update UI adapters and docs; add tests for parsing |
| Change run directory layout or artifact filenames | High | Migration plan; update UI scanner; add regression tests |
| Change promotion gate logic or metric aliasing | High | Regression tests; document changes; update thresholds guidance |
| Change final model training selection rules | High | Integration tests; reviewer-facing documentation updates |

## Testing Requirements
- Unit: `pytest tests/unit/test_project_module.py`
- Integration: `pytest tests/integration/test_project_meta_pipeline.py`
- Integration: `pytest tests/integration/test_final_train_workflow.py`
- Unit: `pytest tests/unit/test_project_backend.py`

## Common Pitfalls
- Schema drift in dataset manifests (missing/renamed columns) causes project runs to fail later; registry should catch this early.
- Inconsistent metric naming between training outputs and thresholds (use `normalize_metric_name`).
- `f1` gates are resolved against available F1 variants (`f1_macro`, then `f1_weighted`, then `f1`)
  so threshold configs remain portable across task modes.
- Assuming calibration is always checked: `brier_calibrated` / `ece_calibrated` are only gating
  when configured under `promotion.calibration`.
- Assuming fold artifacts exist for all modes; binary training outputs differ from meta/multiclass/hierarchical.
- Silent run-id changes due to config hashing changes can “orphan” expected outputs in UI/history.

## Threshold Profiles and Templates
- Bootstrap supports `--gate-profile balanced|f1|sensitivity` to initialize user-facing gate intent.
- Bootstrap supports `--engine`, `--device`, and `--show-options` for mode/engine-aware templates.
- Bootstrap/recommend support `--promotion-gate-template <id>` and
  `--list-promotion-gate-templates`.
- Built-in template IDs:
  - `clinical_conservative`
  - `screen_ruleout`
  - `confirm_rulein`
  - `research_exploratory`
- Recommended baseline profiles:
  - `balanced`: require `f1`/`f1_macro` and `balanced_accuracy`.
  - `f1`: require `f1` (binary) or `f1_macro` (meta/multiclass/hierarchical).
  - `sensitivity`: require only `sensitivity` (recall-first screening workflows).
- Users retain full control by editing `registry/thresholds.yaml` directly after bootstrap,
  including explicit `promotion_gates` overrides.

## Examples
```bash
classiflow project init --name "CNS embryonal" --out projects
classiflow project run-technical projects/PROJECT__cns_embryonal
classiflow project run-independent-test projects/PROJECT__cns_embryonal
```

## High-Risk Change Protocol
- Required design note (ADR):
  - Include filesystem layout diffs, registry schema changes, and promotion gate logic changes.
- Required test additions:
  - Add fixture projects/runs for adapter and scanner parsing.
  - Add regression tests for promotion decisions on fixed metric inputs.
- Required backward compatibility checks:
  - Existing projects must remain readable by UI API (or require explicit migration command).
  - Existing runs must still be bundleable and inferable.
- Required release note items:
  - “Project workflow changes” including any required manual migration steps.
