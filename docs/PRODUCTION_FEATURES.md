# Production & Publication Features Guide

This guide covers the enhanced features added in v0.2+ for production deployment and publication-ready outputs.

## Table of Contents

- [Model + Data Lineage](#model--data-lineage)
- [Portable Model Bundles](#portable-model-bundles)
- [Inference Confidence Metrics](#inference-confidence-metrics)
- [Feature Drift Detection](#feature-drift-detection)
- [Hierarchical Evaluation Metrics](#hierarchical-evaluation-metrics)
- [Migration Guide](#migration-guide)

---

## Model + Data Lineage

### Overview

Every training and inference run now includes full provenance tracking for reproducibility and auditability.

### Training Run Lineage

Training automatically generates `run.json` with:

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-13T12:00:00",
  "package_version": "0.2.0",
  "training_data_path": "data/training.csv",
  "training_data_hash": "abc123...",
  "training_data_size_bytes": 1048576,
  "training_data_row_count": 1000,
  "feature_list": ["feature1", "feature2", ...],
  "config": {...}
}
```

### Inference Run Lineage

Inference creates `inference_run.json` linked to parent training:

```json
{
  "inference_run_id": "660e8400-e29b-41d4-a716-446655440001",
  "parent_run_id": "550e8400-e29b-41d4-a716-446655440000",
  "inference_data_hash": "def456...",
  "drift_warnings": [...],
  "validation_warnings": [...]
}
```

### Predictions Include Lineage

Every row in `predictions.csv` includes:

```csv
sample_id,predicted_label,model_run_id,inference_run_id,inference_data_hash
sample_001,ClassA,550e8400-...,660e8400-...,def456...
```

### API Usage

```python
from classiflow.lineage import create_training_manifest, get_file_metadata

# During training
data_metadata = get_file_metadata(Path("data.csv"))
manifest = create_training_manifest(
    data_path=Path("data.csv"),
    data_hash=data_metadata["sha256_hash"],
    data_size_bytes=data_metadata["size_bytes"],
    data_row_count=data_metadata["row_count"],
    config=config.to_dict(),
    feature_list=feature_names,
)
manifest.save(outdir / "run.json")
```

---

## Portable Model Bundles

### Overview

Bundles are self-contained ZIP archives that package everything needed for inference, enabling easy model sharing and offline deployment.

### Creating Bundles

```bash
# Bundle a single fold
classiflow bundle create --run-dir derived/fold1 --out my_model.zip

# Bundle all folds
classiflow bundle create --run-dir derived --out my_model.zip --all-folds

# Add custom description
classiflow bundle create \
    --run-dir derived \
    --out my_model_v1.2.zip \
    --description "Production model v1.2 - trained on cohort A"
```

### Bundle Contents

```
my_model.zip
├── run.json              # Training manifest
├── artifacts.json        # Artifact registry
├── version.txt          # Package version
├── README.txt           # Usage instructions
├── fold1/
│   ├── binary_smote/
│   │   ├── binary_pipes.joblib
│   │   ├── meta_model.joblib
│   │   └── ...
│   └── ...
└── metrics_inner_cv.csv  # (optional)
```

### Inspecting Bundles

```bash
# Basic inspection
classiflow bundle inspect my_model.zip

# Verbose (show all files)
classiflow bundle inspect my_model.zip --verbose
```

Output:
```
======================================================================
Bundle: my_model.zip
======================================================================
Size: 12.34 MB
Files: 47
Valid: ✓

----------------------------------------------------------------------
Training Run
----------------------------------------------------------------------
Run ID: 550e8400-e29b-41d4-a716-446655440000
Task Type: meta
Timestamp: 2024-01-13T12:00:00
Package Version: 0.2.0
Training Data: data/training.csv
Data Hash: abc123...
Data Rows: 1000

----------------------------------------------------------------------
Artifacts
----------------------------------------------------------------------
fold1: 25 file(s)
fold2: 22 file(s)
```

### Using Bundles for Inference

```bash
# Command-line
classiflow infer \
    --bundle my_model.zip \
    --data-csv test_data.csv \
    --outdir results/

# Python API
from classiflow.bundles import load_bundle

bundle_data = load_bundle("my_model.zip", fold=1)
manifest = bundle_data["manifest"]
fold_dir = bundle_data["fold_dir"]
```

### Validating Bundles

```bash
classiflow bundle validate my_model.zip
```

Checks:
- Required files present
- Version compatibility
- Manifest integrity

---

## Inference Confidence Metrics

### Overview

Every prediction includes per-sample uncertainty metrics to help assess prediction reliability.

### Confidence Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `confidence_max_proba` | Maximum probability across classes | [0, 1] |
| `confidence_margin` | Difference between top-1 and top-2 probabilities | [0, 1] |
| `confidence_entropy` | Entropy of probability distribution | [0, log(n_classes)] |
| `confidence_bucket` | Categorical bucket: high/medium/low | categorical |

### Predictions CSV Example

```csv
sample_id,predicted_label,confidence_max_proba,confidence_margin,confidence_entropy,confidence_bucket
001,ClassA,0.95,0.90,0.12,high
002,ClassB,0.82,0.64,0.45,medium
003,ClassA,0.58,0.16,1.02,low
```

### Confidence Buckets

Default thresholds:
- **High**: max_proba ≥ 0.9
- **Medium**: 0.7 ≤ max_proba < 0.9
- **Low**: max_proba < 0.7

Custom thresholds:
```python
from classiflow.inference.confidence import assign_confidence_buckets

buckets = assign_confidence_buckets(
    confidence_scores,
    thresholds={"high_min": 0.8, "medium_min": 0.6, "low_min": 0.0}
)
```

### Filtering by Confidence

```python
from classiflow.inference.confidence import filter_by_confidence

# Keep only high-confidence predictions
high_conf = filter_by_confidence(
    predictions_df,
    min_confidence=0.9,
    confidence_col="confidence_max_proba"
)
```

### Excel Output

The inference Excel workbook includes a `Confidence_Summary` sheet with:
- Distribution statistics (mean, std, quartiles)
- Bucket counts and percentages

---

## Feature Drift Detection

### Overview

Automatic monitoring of feature distribution changes between training and inference data.

### Drift Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `z_shift` | `(mean_inf - mean_train) / std_train` | Standard deviations shifted |
| `missing_delta` | `missing_rate_inf - missing_rate_train` | Change in missing values |
| `median_shift` | `(median_inf - median_train) / IQR_train` | IQR-normalized median shift |

### Detection Thresholds

Default thresholds:
- **z_shift**: ±3.0 (3 standard deviations)
- **missing_delta**: ±0.1 (10% change in missing rate)
- **median_shift**: ±2.0 (2 IQRs)

### CLI Usage

```bash
classiflow infer \
    --data-csv test_data.csv \
    --run-dir derived/fold1 \
    --outdir results/

# Generates:
# - feature_drift_summary.csv
# - feature_drift_summary.xlsx (with Drift and Drift_Flagged sheets)
# - Drift warnings in inference_run.json
```

### Drift Report Structure

**feature_drift_summary.csv:**
```csv
feature,z_shift,missing_delta,median_shift,mean_train,mean_inference,...
feature1,5.2,0.02,3.1,10.0,20.4,...
feature2,0.3,0.15,0.5,50.0,51.5,...
```

**Excel Sheets:**
- `Drift`: All features with drift scores
- `Drift_Flagged`: Features exceeding thresholds
- `Thresholds`: Detection thresholds used

### Python API

```python
from classiflow.validation import (
    compute_feature_summary,
    compute_drift_scores,
    detect_drift,
)

# Training time
train_summary = compute_feature_summary(X_train, feature_names)
save_feature_summaries(train_summary, outdir / "feature_summaries.json")

# Inference time
inf_summary = compute_feature_summary(X_inf, feature_names)
drift_df = compute_drift_scores(train_summary, inf_summary)
flagged, warnings = detect_drift(drift_df)

# Review flagged features
print(flagged[["feature", "z_shift", "missing_delta"]].head())
```

### Streamlit Integration

The inference page displays:
- Drift warning banner (if features flagged)
- Top 5 drifted features
- Interactive drift visualization

---

## Hierarchical Evaluation Metrics

### Overview

For hierarchical models (e.g., `TumorType::Subtype::Variant`), compute metrics at each level.

### Hierarchical Label Format

```
L1::L2::L3
TumorTypeA::SubtypeX::VariantAlpha
```

### Metrics at Each Level

For each level (L1, L2, L3, Leaf):
- **Overall**: accuracy, macro F1
- **Per-class**: precision, recall, F1, support
- **Confusion matrix**: true vs predicted

### Excel Output Structure

```
inference_results.xlsx
├── Overall_Metrics_L1
├── Per_Class_L1
├── Confusion_L1
├── Overall_Metrics_L2
├── Per_Class_L2
├── Confusion_L2
├── Overall_Metrics_L3
├── Per_Class_L3
├── Confusion_L3
├── Overall_Metrics_Leaf
├── Per_Class_Leaf
└── Confusion_Leaf
```

### Python API

```python
from classiflow.inference.hierarchical_metrics import compute_hierarchical_metrics

metrics = compute_hierarchical_metrics(
    y_true=pd.Series(["TypeA::SubX", "TypeB::SubY", ...]),
    y_pred=pd.Series(["TypeA::SubX", "TypeB::SubZ", ...]),
    y_proba=probabilities,  # optional
)

# Access level-specific metrics
l1_accuracy = metrics["l1"]["accuracy"]
l1_f1_macro = metrics["l1"]["f1_macro"]

# Per-class metrics
for cls_info in metrics["l1"]["per_class"]:
    print(f"{cls_info['class']}: F1={cls_info['f1']:.3f}")
```

### Streamlit Integration

The inference page includes:
- Level selector dropdown
- Switchable confusion matrices
- Level-specific performance tables

---

## Migration Guide

### Backward Compatibility

All new features are **backward compatible**:
- Existing runs without `run.json` use legacy `run_manifest.json`
- Missing `feature_summaries.json` → drift detection skipped
- Confidence metrics always computed (new feature)
- Hierarchical metrics auto-detected from label format

### Enabling Full Lineage for Existing Runs

1. **Re-train** models using updated CLI:
   ```bash
   classiflow train-meta --data-csv data.csv --label-col subtype --outdir derived/
   ```

2. **Feature summaries** are saved automatically at `outdir/feature_summaries.json`

3. **Inference** will now include full lineage tracking

### Legacy Compatibility Table

| Feature | Legacy Run | New Run |
|---------|-----------|---------|
| Basic inference | ✓ | ✓ |
| Lineage tracking | Partial | Full |
| Drift detection | ✗ | ✓ |
| Confidence metrics | ✗ | ✓ |
| Bundle support | ✗ | ✓ |
| Hierarchical metrics | Partial | Full |

### Updating Existing Workflows

**Before:**
```bash
classiflow train-meta --data-csv data.csv --label-col subtype
classiflow infer --data-csv test.csv --run-dir derived/fold1
```

**After (with new features):**
```bash
# Training (auto-generates lineage)
classiflow train-meta --data-csv data.csv --label-col subtype

# Create bundle
classiflow bundle create --run-dir derived --out model_v1.zip

# Inference with bundle
classiflow infer --bundle model_v1.zip --data-csv test.csv --outdir results/

# Inspect results
cat results/inference_run.json  # Check lineage
cat results/feature_drift_summary.csv  # Check drift
```

---

## Best Practices

### For Production Deployment

1. **Always create bundles** for model versioning:
   ```bash
   classiflow bundle create --run-dir derived --out models/model_${VERSION}.zip
   ```

2. **Monitor drift** in production:
   - Set up automated drift monitoring
   - Alert on features exceeding thresholds
   - Retrain models when drift detected

3. **Filter by confidence**:
   - Route low-confidence predictions for manual review
   - Track confidence distribution over time

4. **Maintain lineage**:
   - Store `run_id` and `inference_run_id` in production database
   - Enable tracing predictions back to training data

### For Publications

1. **Include lineage metadata**:
   - Report `run_id`, data hashes, package version
   - Include `run.json` in supplementary materials

2. **Report confidence metrics**:
   - Include confidence distribution plots
   - Report metrics stratified by confidence bucket

3. **Document drift**:
   - Compare training vs. validation/test distributions
   - Report flagged features and mitigation strategies

4. **Hierarchical reporting**:
   - Report metrics at each level
   - Include level-specific confusion matrices

---

## Troubleshooting

### Bundle Creation Fails

**Error**: `FileNotFoundError: No run.json or run_manifest.json`

**Solution**: Ensure training completed successfully and run directory contains manifest.

### Drift Detection Skipped

**Warning**: `Training feature summaries not found`

**Solution**: Re-run training to generate `feature_summaries.json`, or manually create:
```python
from classiflow.validation import compute_feature_summary, save_feature_summaries
summary = compute_feature_summary(X_train, feature_names)
save_feature_summaries(summary, outdir / "feature_summaries.json")
```

### Version Compatibility Warning

**Warning**: `Version mismatch: bundle=0.1.0, current=0.2.0`

**Solution**: Bundles from older versions may work but are not guaranteed. Retrain with current version for best compatibility.

---

## Examples

### Complete Workflow with All Features

```bash
# 1. Train model
classiflow train-meta \
    --data-csv training_data.csv \
    --label-col diagnosis \
    --outdir derived/ \
    --outer-folds 5

# 2. Create bundle
classiflow bundle create \
    --run-dir derived \
    --out models/model_v1.0.zip \
    --all-folds \
    --description "Initial production model"

# 3. Validate bundle
classiflow bundle validate models/model_v1.0.zip

# 4. Run inference with all features
classiflow infer \
    --bundle models/model_v1.0.zip \
    --data-csv test_data.csv \
    --outdir results/ \
    --label-col diagnosis

# 5. Review results
classiflow bundle inspect models/model_v1.0.zip
cat results/inference_run.json
cat results/feature_drift_summary.csv
python -c "import pandas as pd; df = pd.read_csv('results/predictions.csv'); print(df[df['confidence_bucket']=='low'])"
```

---

## API Reference

See individual module documentation:
- [classiflow.lineage](src/classiflow/lineage)
- [classiflow.bundles](src/classiflow/bundles)
- [classiflow.inference.confidence](src/classiflow/inference/confidence.py)
- [classiflow.validation](src/classiflow/validation)
- [classiflow.inference.hierarchical_metrics](src/classiflow/inference/hierarchical_metrics.py)
