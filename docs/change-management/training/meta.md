# training/meta

## Objective
- Train a **meta-classifier** for multiclass problems by:
  - training many binary tasks (OvR/pairwise/composite)
  - generating meta-features from task scores
  - fitting a multiclass classifier (optionally calibrated)

## Public Interfaces
- `train_meta_classifier(config: MetaConfig) -> dict` in `src/classiflow/training/meta.py`

## Inputs
- `MetaConfig` key fields:
  - `data_path`/`data_csv`, `label_col`, optional `patient_col`
  - `classes` (subset/order) and optional `tasks_json` + `tasks_only`
  - `smote_mode` (`off`/`on`/`both`)
  - `backend` (`sklearn`/`torch`) and device settings for base estimators
  - calibration controls:
    - `calibration_enabled` (`false`/`true`/`auto`) and legacy `calibrate_meta` compatibility toggle
    - `calibration_method` (`sigmoid`/`isotonic`), `calibration_cv`, `calibration_bins`, `calibration_binning`
    - `calibration_isotonic_min_samples`
    - policy controls: `calibration_policy_apply_to_modes`, `calibration_policy_force_keep`,
      `calibration_policy_thresholds` (R1-R4 guardrails)

## Outputs
- Training manifest:
  - `outdir/run.json`
- Metrics tables (top-level):
  - `outdir/metrics_inner_cv.csv`
  - `outdir/metrics_inner_cv_splits.csv` and `.xlsx`
  - `outdir/metrics_outer_binary_eval.csv`
  - `outdir/metrics_outer_meta_eval.csv`
    - includes promotion-facing decision metrics for val rows:
      `sensitivity`, `specificity`, `ppv`, `npv`, `recall`, `precision`, `mcc`
- Binary learner diagnostics (top-level and fold-level, additive):
  - `outdir/binary_learners_manifest.json`
  - `outdir/binary_learners_metrics_by_fold.csv`
  - `outdir/binary_learners_metrics_summary.csv`
  - `outdir/binary_learners_warnings.json`
  - `outdir/ovo_auc_matrix.csv`
  - `outdir/plots/binary_ovr_roc_<class>.png`
  - `outdir/plots/binary_ovr_roc_all_classes.png`
  - `outdir/plots/ovo_auc_matrix.png`
- Fold artifacts (per outer fold and variant):
  - `outdir/fold{N}/binary_{variant}/binary_pipes.joblib` (dict with `pipes` and `best_models`)
  - `outdir/fold{N}/binary_{variant}/base_ovr_proba_fold{N}.npz` (base OVR probabilities with `sample_id`, `y_true`, class order)
  - `outdir/fold{N}/binary_{variant}/meta_model.joblib`
  - `outdir/fold{N}/binary_{variant}/meta_features.csv` (ordered column names)
  - `outdir/fold{N}/binary_{variant}/meta_classes.csv` (ordered class names)
  - `outdir/fold{N}/binary_{variant}/calibration_metadata.json`
  - `outdir/fold{N}/binary_{variant}/calibration_summary.json`
  - `outdir/fold{N}/binary_{variant}/calibration_curve.csv` (top1 alias path for compatibility)
  - `outdir/fold{N}/binary_{variant}/calibration_curve_<curve_name>.csv` (e.g., `top1`, `binary_pos`, `ovr_*`)
  - `outdir/fold{N}/binary_{variant}/calibration_curve_<curve_name>_uncalibrated.csv` (diagnostic)
  - `outdir/fold{N}/binary_{variant}/calibration_curve_<curve_name>_calibrated.csv` (diagnostic)
  - explicit top1 aliases:
    - `calibration_curve_top1_uncalibrated.csv`
    - `calibration_curve_top1_calibrated.csv`
  - `outdir/fold{N}/binary_{variant}/threshold_config.json`
- Plots:
  - per-fold and averaged ROC/PR and confusion matrix outputs (see module for filenames)

## Internal Workflow
- Build tasks via `TaskBuilder` and optional `load_composite_tasks`.
- Outer CV:
  - stratified group CV when `patient_col` provided (with leakage checks)
  - else `StratifiedKFold`
- For each fold and variant:
  - fit best binary pipelines per task via inner CV grid search
  - save binary pipelines (`binary_pipes.joblib`)
  - build meta-feature matrices:
    - train: OOF scores for task-member rows; no in-sample fallback
    - val: direct scores for **all** rows (must not depend on validation labels)
  - drop rows with missing OOF scores (no in-sample fallback)
  - train meta estimator(s) via grid search
  - compute uncalibrated and candidate calibrated probability-quality metrics on identical
    validation rows
  - apply policy decision (`R1`-`R4`) and set final variant (`calibrated`/`uncalibrated`)
  - save calibration metadata, decision payload, and both curve variants
  - persist base OVR probabilities for under-the-hood diagnostics
- After outer CV:
  - compute class-level binary learner health metrics/warnings (`BL-001`..`BL-006`)
  - generate OVR ROC plots per class + OVO AUC matrix from final probabilities
  - attach additive diagnostics pointers under `run.json :: artifact_registry.binary_learners`

## Dependencies
- Tasks: `classiflow.tasks.*`
- Splitting: `classiflow.splitting.*` (patient-safe CV)
- Models/backends: `classiflow.backends.registry.get_model_set`
- Metrics: `classiflow.metrics.*` and `classiflow.metrics.scorers.SCORER_ORDER`
- Lineage: `classiflow.lineage.*` for `run.json`

## Invariants & Safety Constraints
- Meta-feature column order is a public contract:
  - `meta_features.csv` must align with how inference constructs `X_meta` (`inference/predict.MetaPredictor`).
- Calibration artifacts must remain consistent:
  - `calibration_metadata.json` and `calibration_curve.csv` are used for reporting and validation.
  - Additional `calibration_curve_<curve_name>.csv` files are additive and mode-dependent.
  - `calibration_summary.json` now includes both calibrated/uncalibrated metrics and
    `overall.probability_quality.calibration_decision`.
- Leakage prevention for `patient_col` must remain enforced for both outer and inner splits.
- OOF scoring must not fall back to in-sample predictions; missing OOF rows are dropped.
- Outer validation meta-feature construction must be label-agnostic:
  - do not use `y_va` to decide which task score columns are populated.
  - validation scoring semantics must match inference (`BinaryPredictor` scores all samples).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new task families (additive) | Medium | Update tests; ensure meta feature naming/order remains stable |
| Change task score definitions or OOF generation | High | Regression tests for inference consistency and calibration |
| Change artifact filenames/layout under `fold{N}/binary_{variant}/` | High | Bundle/inference/UI compatibility tests; migration guidance |

## Testing Requirements
- Unit: `pytest tests/inference/test_meta_predictor.py`
- Unit: `pytest tests/training/test_meta_oof_no_fallback.py`
- Integration: `pytest tests/integration/test_meta_inference_consistency.py`
- Integration: `pytest tests/integration/test_meta_class_alignment_deep.py`
- Training: `pytest tests/training/test_meta_calibration.py`

## Common Pitfalls
- Misalignment between `meta_classes.csv` and `meta_model.classes_` (must remain consistent).
- Including tasks that produce all-NaN labels for a fold (must handle gracefully).
- Calibration on too few samples (isotonic requires minimum sample size; module has controls).

## Examples
```bash
classiflow train-meta --data data.parquet --label-col diagnosis --smote both --outdir derived
```
