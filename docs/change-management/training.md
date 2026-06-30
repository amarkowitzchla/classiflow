# training

## Objective
- Implement the core **model training pipelines** for binary, multiclass, meta-classifier, and hierarchical modes.
- Enforce **nested cross-validation** (outer evaluation + inner hyperparameter tuning) where applicable.
- Persist reviewer-facing artifacts (metrics, plots, manifests) and (for some modes) inference-ready model files.

## Public Interfaces
- Training entrypoints (exported from `src/classiflow/training/__init__.py`):
  - `train_binary_task(config: TrainConfig) -> dict` (`src/classiflow/training/binary.py`)
  - `train_meta_classifier(config: MetaConfig) -> dict` (`src/classiflow/training/meta.py`)
  - `train_multiclass_classifier(config: MulticlassConfig) -> dict` (`src/classiflow/training/multiclass.py`)
  - `NestedCVOrchestrator(...)` (`src/classiflow/training/nested_cv.py`)
- Hierarchical training (called by CLI, not re-exported in `training/__init__.py`):
  - `train_hierarchical(config: HierarchicalConfig) -> dict` (`src/classiflow/training/hierarchical_cv.py`)
- CLI entrypoints (Typer):
  - `classiflow train-binary`, `train-meta`, `train-multiclass`, `train-hierarchical` (`src/classiflow/cli/main.py`)

Sub-docs:
- `docs/change-management/training/nested_cv.md`
- `docs/change-management/training/meta.md`
- `docs/change-management/training/multiclass.md`
- `docs/change-management/training/hierarchical_cv.md`

## Inputs
- Data inputs:
  - `TrainConfig.resolved_data_path` / `MetaConfig.resolved_data_path` / `MulticlassConfig.resolved_data_path`
  - supports CSV, Parquet, Parquet dataset directories (via `classiflow.data.load_table`)
- Leakage controls:
  - patient/group stratification enabled when `patient_col` is provided (binary/meta/multiclass)
- Randomness controls:
  - `random_state` (must drive splitting and estimator seeds)
- Backend/device:
  - binary/meta support `backend="sklearn"|"torch"` via `backends/registry.get_model_set`
  - multiclass uses `resolve_device` and can include torch estimators depending on `estimator_mode`
- Meta-specific:
  - `classes` and optional `tasks_json`/`tasks_only`
  - calibration controls (`calibration_enabled`, `calibration_method`, `calibration_cv`,
    `calibration_bins`, policy thresholds; legacy `calibrate_meta` compatibility retained)
- Hierarchical-specific:
  - `label_l1`, optional `label_l2` to enable hierarchical mode
  - `use_smote` and torch MLP hyperparameters

## Outputs
- Common outputs across modes:
  - `run.json` training manifest (via `classiflow.lineage.manifest.create_training_manifest`)
  - metrics CSVs and fold plots under `outdir`
  - `artifact_registry.probability_quality` fold payloads in `run.json` (final variant + per-variant metrics when available)
  - compatibility calibration curve path `calibration_curve.csv` (top1 final variant) plus mode/variant-tagged curve files
- Mode-specific outputs:
  - **Binary (`train_binary_task`)**: metrics + plots; does not currently persist inference-ready pipelines.
  - **Meta (`train_meta_classifier`)**: persists fold artifacts under `fold{N}/binary_{variant}/` (see sub-doc).
  - **Multiclass (`train_multiclass_classifier`)**: persists `multiclass_model.joblib`, `classes.csv`, `feature_list.csv` per fold/variant.
  - **Hierarchical (`train_hierarchical`)**: persists `training_config.json`, `fold*/scaler.joblib`, label encoders, `model_*.pt` files, and `run.json`.

## Internal Workflow
- High-level flow (binary/meta/multiclass):
  - Load data (`io/loaders.py`), validate (`io/loaders.validate_data`)
  - If `patient_col` is set, compute group labels and split using `splitting/` iterators with leakage checks
  - Inner CV grid search using multi-metric scorers (`metrics/scorers.py`) and refit metric
  - Outer fold evaluation and artifact generation (metrics tables, plots)
  - Save manifests and mode-specific artifacts
- Hierarchical flow:
  - Load raw table (`data/loaders.load_table`)
  - Build feature list, fit scaler per fold, tune torch MLP hyperparameters, save `.pt` models and plots

## Dependencies
- Upstream callers:
  - CLI `classiflow train-*` commands
  - Projects orchestration reuses some training functions in workflows
- Downstream calls:
  - `classiflow.io`, `classiflow.data`, `classiflow.splitting`
  - `classiflow.metrics` and `classiflow.plots`
  - `classiflow.backends` and `classiflow.models`
  - `classiflow.lineage` for manifest creation
- External dependencies: `pandas`, `numpy`, `sklearn`, `imblearn`, optional `torch`, `joblib`.

## Invariants & Safety Constraints
- Nested CV correctness:
  - inner CV must be performed using only outer-train data
  - no information from outer-val may influence model selection
- No leakage when `patient_col` is provided:
  - both outer and inner splits must keep patient IDs disjoint between train/val
  - explicit checks exist (`splitting.assert_no_patient_leakage`) and must remain in place
- Determinism:
  - same data + same seeds + same config ⇒ reproducible splits and comparable results
- Artifact/manifest stability:
  - `run.json` fields and per-mode artifact filenames are part of the API used by inference/bundles/projects/UI

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Refactor training code without output changes | Medium | Run targeted training/integration tests; ensure metrics/artifacts unchanged |
| Modify split logic or leakage behavior | High | Add regression tests; document baseline shifts; follow high-risk protocol |
| Change artifact schemas/filenames or manifest contents | High | Migration plan; bundle/inference compatibility tests; update UI adapters |
| Add new estimator families/backends | Medium | Add tests; ensure deterministic seeding and joblib compatibility |

## Testing Requirements
- Training suite:
  - `pytest tests/training/test_patient_stratified_wiring.py`
  - `pytest tests/training/test_meta_calibration.py`
  - `pytest tests/training/test_multiclass.py`
  - `pytest tests/training/test_multiclass_hardening.py`
  - `pytest tests/training/test_torch_backend_binary.py`
- Integration suite:
  - `pytest tests/integration/test_meta_inference_consistency.py`
  - `pytest tests/integration/test_final_train_workflow.py`

## Common Pitfalls
- Leakage via fitting transforms on full data (ensure scalers/pipelines are fit only on training splits).
- Label encoding drift across folds (meta/multiclass must preserve class ordering artifacts).
- Silent changes to “best model” selection metric (refit scorer) that shift results.
- Device fallback changing runtime behavior (torch backend should log and enforce `require_torch_device` when requested).

## Examples
```bash
classiflow train-meta --data data.parquet --label-col diagnosis --patient-col patient_id --smote both --outdir derived
```

## High-Risk Change Protocol
- Required design note (ADR):
  - Describe the change, affected modes (binary/meta/multiclass/hierarchical), and leakage/determinism implications.
  - Include artifact/manifest schema diffs if any.
- Required test additions:
  - Add a regression test covering the pre-change bug and expected behavior.
  - Add a golden/fixture-based test for artifact filenames and key CSV schema columns when changing outputs.
- Required backward compatibility checks:
  - Existing `run.json` and saved model artifacts must still load in inference/bundle/UI, or fail loudly with remediation.
- Required release note items:
  - Expected metrics shifts, migration steps for existing runs, and any new required dependencies.
