# Classiflow UI Adapter Notes

This document describes the on-disk structure and manifest formats discovered during codebase exploration,
serving as the source of truth for the UI API adapter layer.

## Project Structure

Projects live under a configurable `projects_root` directory (e.g., `./projects`).

```
projects/
└── {project_id}__{slugified_name}/    # e.g., MRS_MEDULLO__mrs_medullo
    ├── project.yaml                   # Project configuration (ProjectConfig model)
    ├── README.md                      # Optional project documentation
    ├── data/
    │   ├── train/                     # Training data manifests
    │   └── test/                      # Test data manifests
    ├── registry/
    │   ├── datasets.yaml              # DatasetRegistry - registered datasets with hashes
    │   └── thresholds.yaml            # Promotion thresholds per phase
    ├── promotion/
    │   ├── decision.yaml              # Promotion gate decision (PASS/FAIL + reasons)
    │   └── promotion_report.md        # Human-readable promotion summary
    └── runs/
        ├── technical_validation/      # Phase 1: Nested CV validation
        │   └── {run_id}/
        ├── independent_test/          # Phase 2: Held-out test evaluation
        │   └── {run_id}/
        └── final_model/               # Phase 3: Production model bundle
            └── {run_id}/
```

## Manifest Formats

### Primary: `run.json` (TrainingRunManifest)

Located at: `runs/{phase}/{run_id}/run.json`

Key fields observed:
```json
{
  "run_id": "ae196723-608c-4f7c-8722-66492740e949",  // UUID4
  "timestamp": "2026-01-15T07:15:13.259043",         // ISO timestamp
  "package_version": "0.1.0",
  "training_data_path": "/path/to/data.csv",
  "training_data_hash": "sha256...",
  "training_data_size_bytes": 32542,
  "training_data_row_count": 94,
  "config": {
    "data_csv": "...",
    "label_col": "MOLECULAR",
    "outer_folds": 3,
    "inner_splits": 5,
    "smote_mode": "off",
    // ... other training params
  },
  "task_type": "meta",                              // binary | meta | hierarchical
  "python_version": "3.11.10",
  "hostname": "imac.lan",
  "git_hash": "833f768...",
  "feature_list": ["Ala_conc", "Asp_conc", ...],    // Feature columns
  "task_definitions": {...},                         // OvR + pairwise task mappings
  "hierarchical": false,
  "l1_classes": null,                               // For hierarchical models
  "l2_classes_per_branch": null
}
```

### Lineage: `lineage.json`

Located at: `runs/{phase}/{run_id}/lineage.json`

```json
{
  "phase": "TECHNICAL_VALIDATION",                  // TECHNICAL_VALIDATION | INDEPENDENT_TEST | FINAL_MODEL
  "run_id": "02c3cc46b589",                         // Short hex ID (directory name)
  "timestamp_local": "2026-01-15T07:17:09.010622",
  "timestamp_utc": "2026-01-15T15:17:09.010630+00:00",
  "classiflow_version": "0.1.0",
  "git_hash": "833f768...",
  "python_version": "3.11.10",
  "platform": "macOS-15.7.3-x86_64-i386-64bit",
  "config_hash": "sha256...",
  "dataset_hashes": {
    "train": "sha256...",
    "test": "sha256..."
  },
  "command": "classiflow project run-technical",
  "args": {"run_id": "02c3cc46b589"},
  "outputs": [
    {"path": "...", "sha256": "..."}
  ]
}
```

**Important**: The `run_id` in `lineage.json` is the SHORT hex ID (directory name),
while `run_id` in `run.json` is a full UUID4. Use the directory name as canonical run identifier.

## Metrics Formats

### Phase: technical_validation

Primary: `metrics_summary.json`
```json
{
  "summary": {"balanced_accuracy": 0.857},
  "per_fold": {"balanced_accuracy": [0.819, 0.908, 0.843]}
}
```

Also produces:
- `metrics_inner_cv.csv` - Inner CV results
- `metrics_inner_cv_splits.xlsx` - Detailed inner splits
- `metrics_outer_meta_eval.csv` - Outer fold meta-classifier evaluation
- `metrics_outer_binary_eval.csv` - Outer fold binary task evaluation

### Phase: independent_test

Primary: `metrics.json`
```json
{
  "overall": {
    "n_samples": 94,
    "accuracy": 0.670,
    "balanced_accuracy": 0.594,
    "f1_macro": 0.602,
    "f1_weighted": 0.643,
    "mcc": 0.533,
    "per_class": [
      {"class": "G3", "precision": 0.53, "recall": 0.36, "f1": 0.43, "support": 22},
      ...
    ],
    "confusion_matrix": {"labels": [...], "matrix": [[...]]},
    "roc_auc": {"per_class": [...], "macro": 0.78, "micro": 0.85},
    "log_loss": 1.06
  }
}
```

Also produces:
- `metrics.xlsx` - Excel workbook with all metrics
- `metrics/overall_metrics.csv`
- `metrics/per_class_metrics.csv`
- `metrics/confusion_matrix.csv`
- `metrics/roc_auc_summary.csv`
- `metrics/run_manifest.csv`

### Phase: final_model

May have `run.json` only (copied from training). No specific metrics file.

## Artifact Types by Phase

### technical_validation
- `run.json` - Training manifest
- `lineage.json` - Execution lineage
- `config.resolved.yaml` - Resolved configuration
- `metrics_summary.json` - Summary metrics
- `metrics_*.csv` / `metrics_*.xlsx` - Detailed metrics
- `roc_meta_averaged.png` - Averaged ROC curves
- `pr_meta_averaged.png` - Averaged PR curves
- `reports/technical_validation_report.md` - Generated report
- `fold{N}/binary_none/` - Per-fold artifacts:
  - `meta_model.joblib` - Trained meta-classifier
  - `binary_pipes.joblib` - Binary task pipelines
  - `meta_classes.csv`, `meta_features.csv` - Metadata
  - `cm_meta_fold{N}.png` - Confusion matrix plot
  - `roc_meta_fold{N}.png` - ROC curve plot
  - `pr_meta_fold{N}.png` - PR curve plot

### independent_test
- `lineage.json` - Execution lineage
- `metrics.json` - Evaluation metrics
- `metrics.xlsx` - Excel metrics workbook
- `metrics/*.csv` - Individual metric CSVs
- `predictions.csv` - Model predictions
- `inference_confusion_matrix.png` - Confusion matrix plot
- `inference_roc_curves.png` - ROC curves plot
- `inference_score_distributions.png` - Score distribution plot
- `reports/independent_test_report.md` - Generated report

### final_model
- `run.json` - Training manifest (from technical_validation)
- `lineage.json` - Bundle creation lineage
- `model_bundle.zip` - Portable model bundle
- `fold1/binary_none/` - Production model artifacts:
  - `meta_model.joblib`
  - `binary_pipes.joblib`
  - `meta_classes.csv`, `meta_features.csv`

## Registry Files

### datasets.yaml (DatasetRegistry)
```yaml
datasets:
  train:
    dataset_type: train
    manifest_path: /path/to/data.csv
    sha256: "hash..."
    size_bytes: 32542
    registered_at: "2026-01-15T15:14:47"
    classiflow_version: "0.1.0"
    schema:
      columns: [...]
      dtypes: {...}
      feature_columns: [...]
    stats:
      rows: 94
      labels: {G4: 35, SHH: 26, G3: 22, WNT: 11}
    dirty: false
  test:
    # Similar structure
updated_at: "2026-01-15T15:14:47"
```

### thresholds.yaml
```yaml
technical_validation:
  required:
    f1: 0.7
    balanced_accuracy: 0.7
  stability:
    std_max: {f1: 0.1, balanced_accuracy: 0.1}
    pass_rate_min: 0.8
independent_test:
  required:
    f1_macro: 0.7
    balanced_accuracy: 0.7
promotion_logic: ALL_REQUIRED_AND_STABILITY
promotion:
  calibration:
    brier_max: 0.20
    ece_max: 0.25
override:
  allow_override: true
  require_comment: true
  require_approver: true
```

### decision.yaml (Promotion Gate Result)
```yaml
decision: FAIL                          # PASS | FAIL
timestamp: "2026-01-15T15:26:02"
technical_run: 02c3cc46b589
test_run: 669e8e97d037
reasons:
  - "technical_validation: Missing metric: f1"
  - "independent_test: f1_macro=0.6023 < 0.7000"
override:
  enabled: false
  comment: null
  approver: null
```

`promotion_decision.json` includes the same decision plus explicit gating vs report-only metrics.

## ID Conventions

- **project_id**: Directory name (e.g., `MRS_MEDULLO__mrs_medullo`)
- **phase**: One of `technical_validation`, `independent_test`, `final_model`
- **run_id**: Short hex string from directory name (e.g., `02c3cc46b589`)
- **run_key**: Globally unique composite: `{project_id}:{phase}:{run_id}`
- **artifact_id**: SHA1 hash of `{run_key}:{relative_path}` for stability

## Reusable Helpers

From `classiflow.projects.project_fs`:
- `ProjectPaths` - Convenience paths dataclass
- `slugify()` - Filesystem-safe string conversion
- `project_root()` - Build project root path
- `choose_project_id()` - Generate project ID

From `classiflow.lineage.manifest`:
- `TrainingRunManifest` - Training manifest dataclass
- `InferenceRunManifest` - Inference manifest dataclass
- `load_training_manifest()` - Load manifest with fallback to legacy name

From `classiflow.projects.yaml_utils`:
- `load_yaml()` - Safe YAML loading
- `dump_yaml()` - Consistent YAML dumping

## Compatibility Notes

1. **Missing run.json**: independent_test phase may not have `run.json`.
   Fall back to `lineage.json` for run metadata + `metrics.json` for metrics.

2. **Legacy run_manifest.json**: Some older runs may have `run_manifest.json`
   instead of `run.json`. The `load_training_manifest()` function handles this.

3. **Variable metrics locations**: Metrics may be at run root (`metrics.json`)
   or in subdirectory (`metrics/*.csv`). Scanner should check both.

4. **Nested fold artifacts**: Fold directories follow pattern `fold{N}/binary_none/`
   or `fold{N}/` depending on task type.
