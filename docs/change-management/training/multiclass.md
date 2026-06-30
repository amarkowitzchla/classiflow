# training/multiclass

## Objective
- Train a direct multiclass classifier under nested CV and optionally compare SMOTE vs no-SMOTE variants.
- Persist inference-ready multiclass model artifacts per fold and sampler variant.

## Public Interfaces
- `train_multiclass_classifier(config: MulticlassConfig) -> dict` in `src/classiflow/training/multiclass.py`

## Inputs
- `MulticlassConfig` fields used heavily:
  - `data_path`/`data_csv`, `label_col`, optional `patient_col`
  - `classes` subset/order
  - `smote_mode`
  - `device` and `estimator_mode` (`all`/`torch_only`/`cpu_only`)
  - `group_stratify` controls stratified group split usage
  - logistic regression tuning fields (`logreg_*`)

## Outputs
- Training manifest:
  - `outdir/run.json`
- Per-fold/per-variant artifacts:
  - `outdir/fold{N}/multiclass_{variant}/multiclass_model.joblib`
  - `outdir/fold{N}/multiclass_{variant}/classes.csv`
  - `outdir/fold{N}/multiclass_{variant}/feature_list.csv`
- Metrics tables (top-level):
  - `metrics_inner_cv.csv`
  - `metrics_inner_cv_splits.csv`
  - `metrics_outer_multiclass_eval.csv`
    - includes core + decision-facing columns used by promotion gates:
      `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`,
      `sensitivity`, `specificity`, `ppv`, `npv`, `recall`, `precision`, `mcc`
  - additional CSVs (`inner_cv_results.csv`, `outer_results.csv`) for diagnostic purposes
- Plots:
  - per-fold and averaged ROC/PR curves; confusion matrices

## Internal Workflow
- Load and validate data; encode labels using a categorical with ordered `classes`.
- Resolve device and optionally include torch estimators.
- Outer CV:
  - group-aware with leakage checks when `patient_col` provided
  - else stratified k-fold
- Inner CV:
  - repeated stratified k-fold or group-aware inner splits
- For each variant:
  - build pipeline with scaler and optional `AdaptiveSMOTE`
  - grid search over estimators; refit on `REFIT_SCORER = "F1 Macro"`
  - persist best estimator and its metadata (classes/features)

## Dependencies
- Data/IO: `classiflow.io.*` and `classiflow.data.load_table` (via io)
- Splitting: `classiflow.splitting.*`
- Models: `classiflow.models.*` (estimators, grids, device resolution)
- Plots: `classiflow.plots.*`
- Lineage: `classiflow.lineage.*` for `run.json`

## Invariants & Safety Constraints
- Class order stability:
  - `classes.csv` must reflect the global `classes` order used for encoding.
- Feature order stability:
  - `feature_list.csv` must match the training `X` column order and inference alignment logic.
- Leakage prevention under `patient_col` must be enforced.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new multiclass estimators (additive) | Medium | Add tests for estimator filtering and artifact serialization |
| Change label encoding / class ordering rules | High | Regression tests; update inference compatibility checks |
| Change artifact filenames or directory layout | High | Update bundle/inference/UI loaders; migration guidance |

## Testing Requirements
- Training: `pytest tests/training/test_multiclass.py`
- Training: `pytest tests/training/test_multiclass_hardening.py`
- Training: `pytest tests/training/test_torch_multiclass_estimators.py`

## Common Pitfalls
- Missing classes in some folds (especially with group splits); code logs coverage—keep these checks.
- Misaligned project runtime settings: for project workflows, multiclass backend is now explicit via
  `execution.engine` + `multiclass.backend`; avoid legacy assumptions about `backend: sklearn` + torch toggles.
- ROC/PR curves require per-class probability columns aligned to class order.

## Examples
```bash
classiflow train-multiclass --data data.parquet --label-col diagnosis --patient-col patient_id --smote both --outdir derived
```
