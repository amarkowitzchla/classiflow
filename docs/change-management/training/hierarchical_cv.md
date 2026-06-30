# training/hierarchical_cv

## Objective
- Train a hierarchical classifier (L1 → optional branch-specific L2) under nested CV with optional patient-level stratification.
- Persist torch MLP models and fold-level artifacts suitable for downstream inference and review.

## Public Interfaces
- `train_hierarchical(config: HierarchicalConfig) -> dict` in `src/classiflow/training/hierarchical_cv.py`
- Tuning helper:
  - `get_hyperparam_candidates(base_hidden: int, base_epochs: int) -> list[dict]`

## Inputs
- `HierarchicalConfig` includes:
  - `data_path`/`data_csv`, optional `patient_col`
  - `label_l1`, optional `label_l2` (enables hierarchical mode)
  - optional `l2_classes` filter and `min_l2_classes_per_branch`
  - `feature_cols` optional, else numeric auto-detect excluding labels/patient
    - CV settings and torch MLP settings (device, epochs, batch size, dropout, early stopping)
  - `use_smote` and `smote_k_neighbors`
- Data is loaded via `classiflow.data.load_table`, then filtered/dropped for required columns.

## Outputs
- Config persistence:
  - `outdir/training_config.json`
- Per-fold artifacts (in `outdir/fold{N}/`):
  - `scaler.joblib`
  - `label_encoder_l1.joblib`
  - `model_level1_fold{N}.pt`
  - split export CSVs:
    - `patient_split_fold{N}.csv` (patient-level) or `sample_split_fold{N}.csv` (sample-level)
  - plots:
    - `roc_level1_fold{N}.png`, `pr_level1_fold{N}.png`, `cm_level1_fold{N}.png`
    - feature importance plots when enabled: `feature_importance_l1_fold{N}.png`
  - hierarchical mode adds per-branch artifacts:
    - `label_encoder_l2_<branch>.joblib`
    - `model_level2_<branch>_fold{N}.pt`
    - `roc_level2_<branch>_fold{N}.png`, `pr_level2_<branch>_fold{N}.png`, `cm_level2_<branch>_fold{N}.png`
- Metrics tables (top-level, written as both CSV and Excel):
  - `metrics_inner_cv.*`
  - `metrics_outer_eval.*`
  - `metrics_summary.*`
  - `metrics_outer_eval.*` rows include additive `mcc` for `L1`, `L2_oracle_*`, and `pipeline` levels
    when those levels are present
  - averaged plots: `roc_level1_averaged.png`, `pr_level1_averaged.png` and per-branch averaged curves

## Internal Workflow
- Determine stratification mode:
  - patient-level: enforce a single L1 label per patient (fail closed on conflicts)
  - sample-level: stratify directly on samples
- Outer CV uses `StratifiedShuffleSplit` with fixed test_size (0.2).
- Inner CV uses patient-aware splits when `patient_col` is provided (no patient overlap).
- Early stopping uses a split drawn **only from outer-train**; outer-val is evaluation-only.
- Save scaler and label encoders per fold; train torch MLPs via `TorchMLPWrapper`.

## Dependencies
- Torch MLP: `classiflow.models.torch_mlp.TorchMLPWrapper`
- SMOTE: `classiflow.models.smote.apply_smote`
- Plots: `classiflow.plots.*`
- External deps: `torch`, `sklearn`, `joblib`, `tqdm`, `pandas`, `numpy`.

## Invariants & Safety Constraints
- Patient stratification must not leak patients across outer train/val or inner CV.
- Saved `.pt` models and `.joblib` scalers/encoders are part of the inference contract (`inference/hierarchical.py` and `HierarchicalInference`).
- Branch training must be skipped safely when insufficient L2 classes exist (`min_l2_classes_per_branch`).
- L2 metrics are oracle-gated by true L1 label and must be labeled as such in outputs.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Change fold split strategy or test_size | High | Regression tests; document baseline shifts; update inference expectations |
| Change model file naming or encoder/scaler persistence | High | Update inference loaders; bundle compatibility checks |
| Add deeper hierarchy levels | High | Requires new manifest/artifact schemas and inference routing logic |

## Testing Requirements
- No dedicated hierarchical training tests currently; changes should be guarded via:
  - `pytest tests/unit/test_compatibility.py` (hierarchical compatibility checks)
  - UI/inference tests if hierarchical inference is used in this repo

## Common Pitfalls
- Confusing this hierarchical training pipeline with `splitting/` group-aware CV; this module uses its own stratification mechanism.
- Feature column auto-detection excludes non-numeric columns; label columns must be explicitly excluded when `feature_cols` provided.
- Torch device availability differences across machines; keep `device="auto"` behavior stable and logged.

## Examples
```bash
classiflow train-hierarchical --data data.parquet --patient-col patient_id --label-l1 tumor_type --label-l2 subtype --outdir derived_hierarchical
```
