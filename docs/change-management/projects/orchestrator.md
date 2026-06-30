# projects/orchestrator

## Objective
- Implement the end-to-end project phase orchestration: technical validation, feasibility, final model build, and independent test.

## Public Interfaces
- In `src/classiflow/projects/orchestrator.py` (exported via `projects/__init__.py`):
  - `run_technical_validation(paths: ProjectPaths, config: ProjectConfig, run_id: str|None = None, compare_smote: bool = False) -> Path`
  - `run_feasibility(paths: ProjectPaths, config: ProjectConfig, run_id: str|None = None, ...) -> Path`
  - `build_final_model(paths: ProjectPaths, config: ProjectConfig, ...) -> Path`
  - `run_independent_test(paths: ProjectPaths, config: ProjectConfig, ...) -> Path`

## Inputs
- `ProjectPaths` defines where to read/write:
  - `project.yaml`, `registry/datasets.yaml`, `registry/thresholds.yaml`, run output directories
- Registered datasets:
  - manifest CSV paths resolved relative to project root when needed
  - dataset hash verification via `verify_manifest_hash` depending on config
- Project configuration determines:
  - task mode (`binary`/`meta`/`multiclass`/`hierarchical`)
  - for meta mode: optional `task.tasks_json` and `task.tasks_only` task-construction controls
  - CV settings (`validation.nested_cv`)
  - execution runtime settings from `execution.*` (legacy backend keys normalized at load time)
  - promotion thresholds and calibration requirements

## Outputs
- Writes per-run directories under:
  - `runs/technical_validation/<run_id>/...`
  - `runs/feasibility/<run_id>/...`
  - `runs/independent_test/<run_id>/...`
  - `runs/final_model/<run_id>/...`
- Produces structured JSON for traceability:
  - `lineage.json` containing hashes, args, version, and output hashes (see `_lineage_payload`)
- Produces metrics JSON summaries for UI/promotion:
  - `metrics.json` written in technical and test phases
- Produces technical markdown diagnostics:
  - `reports/technical_validation_report.md` now includes probability-quality checks
    (ECE/Brier rules with artifact evidence pointers and recommendations)
  - meta runs additionally include a `Binary Learner Health Report (Meta Under-the-Hood)`
    section sourced from `binary_learners_*` artifacts and OVR/OVO plots

## Internal Workflow
- Run identity:
  - compute `config_hash` from normalized config dict
  - compute dataset hashes
  - derive `run_id` from phase + hashes (short SHA prefix)
- Technical validation:
  - train using appropriate training entrypoint
  - collect validation metrics (by phase/variant)
  - optionally run SMOTE comparison and persist results
  - write technical report artifacts (`projects/reporting.py`) including
    rule-based probability-quality diagnostics sourced from `run.json` and
    calibration curve CSV artifacts
  - for meta mode, include under-the-hood base learner diagnostics (BL-001..BL-006,
    class summary table, and plot links) from training-emitted artifacts
- Final model:
  - extract selected configurations from technical run (`projects/final_train.py`)
  - train from scratch on full training data
  - run sanity checks and write `sanity_checks.json`
  - build bundle for deployment
- Independent test:
  - run inference using saved final model/bundle or chosen run artifacts
  - compute test metrics and evaluate promotion gates

## Dependencies
- Training: `classiflow.training.*` and `training/hierarchical_cv.train_hierarchical`
- Inference: `classiflow.inference.run_inference` + `InferenceConfig`
- Promotion: `projects/promotion.py`
- Final training: `projects/final_train.py`
- Hashing: `lineage/hashing.compute_file_hash` and dataset registry verification

## Invariants & Safety Constraints
- Run directories must be self-contained and discoverable by UI scanners.
- `lineage.json` and `metrics.json` must remain parseable and stable enough for audits.
- Final training must not reuse fold pipelines implicitly (explicitly “train from scratch”).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Change run_id hashing inputs | High | Migration for existing runs; update UI scanner assumptions |
| Change which artifacts are written in each phase | High | Update UI and documentation; add integration tests |
| Change metric summarization logic | High | Regression tests for promotion decisions and UI metrics display |

## Testing Requirements
- Integration: `pytest tests/integration/test_final_train_workflow.py`
- Integration: `pytest tests/integration/test_project_meta_pipeline.py`
- Integration: `pytest tests/integration/test_technical_validation_report_probability_quality.py`

## Common Pitfalls
- Relative path resolution bugs (manifest paths must be resolved relative to project root).
- Silent overwrites of runs when `run_id` collisions occur (hash inputs must remain stable).
- Divergence between technical validation artifact layout and inference/bundle expectations.

## Examples
```bash
classiflow project run-technical <project_dir> --compare-smote
classiflow project build-bundle <project_dir>
```
