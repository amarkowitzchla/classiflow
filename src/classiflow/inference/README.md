# Inference Pipeline

Production-quality inference service for the classiflow package. Apply trained models to new data with automatic feature alignment, comprehensive metrics, and publication-ready outputs.

## Features

✅ **Multi-Model Support**
- Binary task models (OvR, pairwise, composite)
- Meta-classifiers (binary → multiclass)
- Hierarchical models (L1 → L2 → L3 routing)
- Automatic run-type detection

✅ **Robust Data Handling**
- Strict mode: Fail if required features missing
- Lenient mode: Fill missing features with zero/median
- Automatic type coercion & NaN handling
- Data quality validation warnings

✅ **Comprehensive Evaluation**
- Accuracy, balanced accuracy, F1 (macro/weighted/micro)
- Matthews Correlation Coefficient (MCC)
- ROC AUC (OvR, macro, micro)
- Per-class metrics (precision, recall, F1, support)
- Confusion matrices
- Log loss

✅ **Publication-Ready Outputs**
- `predictions.csv`: Row-level predictions with scores/probabilities
- `metrics.xlsx`: Multi-sheet Excel workbook
- `metrics/`: CSV versions of all sheets
- High-resolution plots (ROC curves, confusion matrices, score distributions)

## Quick Start

### Python API

```python
from classiflow.inference import run_inference, InferenceConfig

# Create configuration
config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="test_data.csv",
    output_dir="inference_results",
    label_col="diagnosis",  # Optional: for evaluation
    id_col="sample_id",     # Optional: sample identifier
)

# Run inference
results = run_inference(config)

# Access results
predictions = results["predictions"]  # DataFrame
metrics = results["metrics"]          # Dict (if labels provided)
output_files = results["output_files"]  # Dict of file paths
```

### CLI

```bash
# Basic inference (no evaluation)
classiflow infer \
  --data-csv test.csv \
  --run-dir derived/fold1 \
  --outdir results

# With evaluation (requires ground-truth labels)
classiflow infer \
  --data-csv test.csv \
  --run-dir derived/fold1 \
  --outdir results \
  --label-col diagnosis \
  --id-col patient_id

# Lenient mode (fill missing features)
classiflow infer \
  --data-csv test.csv \
  --run-dir derived/fold1 \
  --outdir results \
  --lenient \
  --fill-strategy median

# Hierarchical models with GPU
classiflow infer \
  --data-csv test.csv \
  --run-dir derived_hierarchical/fold1 \
  --outdir results \
  --device cuda
```

### Streamlit UI

```bash
streamlit run src/classiflow/streamlit_app/app.py
```

Navigate to **06_Inference** page for interactive inference with:
- Model selection (standard or hierarchical)
- CSV upload with preview
- Configuration options
- Results dashboard with download buttons

## Module Structure

```
src/classiflow/inference/
├── __init__.py         # Public API exports
├── api.py              # Main entry point: run_inference()
├── config.py           # InferenceConfig & RunManifest dataclasses
├── loader.py           # ArtifactLoader for trained models
├── preprocess.py       # FeatureAligner for data validation
├── predict.py          # Prediction engines (binary/meta/hierarchical)
├── metrics.py          # Classification metrics & ROC/AUC
├── plots.py            # ROC curves, confusion matrices
├── reports.py          # Excel/CSV report generation
└── hierarchical.py     # Hierarchical inference (existing)
```

## Configuration Options

### InferenceConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_dir` | Path | **required** | Directory with trained model artifacts |
| `data_csv` | Path | **required** | Input CSV with features |
| `output_dir` | Path | **required** | Output directory for results |
| `id_col` | str | None | Sample ID column name |
| `label_col` | str | None | Ground-truth label column (for evaluation) |
| `strict_features` | bool | True | Fail if features missing; False to fill |
| `lenient_fill_strategy` | str | "zero" | Fill strategy: "zero" or "median" |
| `max_roc_curves` | int | 10 | Max per-class ROC curves to plot |
| `hierarchical_output` | bool | True | Include L1/L2/L3 outputs |
| `include_plots` | bool | True | Generate plots |
| `include_excel` | bool | True | Generate Excel workbook |
| `device` | str | "auto" | Device for PyTorch models |
| `batch_size` | int | 512 | Batch size for inference |
| `verbose` | int | 1 | Verbosity level (0=minimal, 1=standard, 2=detailed) |

## Output Files

After running inference, the output directory contains:

```
inference_results/
├── predictions.csv                      # Row-level predictions
├── metrics.xlsx                         # Multi-sheet workbook (if labels provided)
├── metrics/                             # CSV versions of metrics
│   ├── run_manifest.csv
│   ├── overall_metrics.csv
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   └── roc_auc_summary.csv
├── inference_confusion_matrix.png       # Confusion matrix plot
├── inference_roc_curves.png             # ROC curves (all classes)
└── inference_score_distributions.png    # Score distributions by class
```

### predictions.csv Columns

**Metadata:**
- `id`: Sample identifier (if provided)
- `true_label`: Ground-truth label (if provided)

**Binary Tasks (if applicable):**
- `{task}_score`: Continuous score for each task
- `{task}_pred`: Binary prediction for each task

**Meta-Classifier (if applicable):**
- `predicted_label`: Final predicted class
- `predicted_proba_{class}`: Probability for each class
- `predicted_proba`: Maximum probability

**Hierarchical (if applicable):**
- `predicted_label_L1`: Level-1 prediction
- `predicted_label_L2`: Level-2 prediction
- `predicted_label_L3`: Level-3 prediction (if exists)
- `predicted_label`: Combined prediction (e.g., "TypeA::SubtypeX")
- `predicted_proba_L1_{class}`: L1 probabilities
- `predicted_proba_L2_{class}`: L2 probabilities

### metrics.xlsx Sheets

1. **Run_Manifest**: Run metadata, config, class counts
2. **Overall_Metrics**: Accuracy, F1, ROC AUC, MCC, log loss
3. **Per_Class_Metrics**: Precision, recall, F1, support per class
4. **Confusion_Matrix**: True vs. predicted labels
5. **ROC_AUC_Summary**: Per-class AUC scores
6. **Task_Level_Metrics**: Binary task metrics (if applicable)
7. **L1_Overall/L2_Overall/Pipeline_Overall**: Hierarchical metrics (if applicable)

## Feature Alignment

### Strict Mode (Default)

```python
config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="test.csv",
    output_dir="results",
    strict_features=True,  # Fail if features missing
)
```

**Behavior:**
- Validates all required features present
- Fails with clear error message if features missing
- Recommended for production inference

### Lenient Mode

```python
config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="test.csv",
    output_dir="results",
    strict_features=False,
    lenient_fill_strategy="median",  # Or "zero"
)
```

**Behavior:**
- Fills missing features with zeros or training median
- Logs warnings for each missing feature
- Useful for exploratory inference or incomplete data

## Handling Different Model Types

### Binary Task Models

```python
# Loads binary_pipes.joblib from run directory
# Runs predictions for all tasks
# Outputs: {task}_score and {task}_pred columns
```

### Meta-Classifiers

```python
# Loads binary_pipes.joblib + meta_model.joblib
# Runs binary tasks first, then meta-classifier
# Outputs: predicted_label, predicted_proba_{class}
```

### Hierarchical Models

```python
# Loads L1 and L2 models (PyTorch-based)
# Routes samples: L1 → appropriate L2 branch
# Outputs: predicted_label_L1, predicted_label_L2, predicted_label
```

## Evaluation Metrics

When ground-truth labels are provided (`label_col` parameter), the pipeline computes:

### Overall Metrics
- **Accuracy**: Proportion of correct predictions
- **Balanced Accuracy**: Average of per-class accuracies
- **F1 Scores**: Macro, weighted, and micro averages
- **MCC**: Matthews Correlation Coefficient
- **Log Loss**: Probabilistic loss (if probabilities available)

### ROC AUC
- **Per-Class**: One-vs-Rest AUC for each class
- **Macro**: Average of per-class AUCs
- **Micro**: AUC of aggregated predictions (multiclass only)
- **Handling**: Reports NA for classes absent in test set

### Per-Class Metrics
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in true class

### Confusion Matrix
- True labels (rows) vs. predicted labels (columns)
- Normalized by true class (proportions)
- Also available as raw counts

## Examples

### Example 1: Basic Binary Classifier

```python
from classiflow.inference import run_inference, InferenceConfig

config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="test_binary.csv",
    output_dir="binary_results",
)

results = run_inference(config)
print(f"Processed {len(results['predictions'])} samples")
```

### Example 2: Meta-Classifier with Evaluation

```python
config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="test_multiclass.csv",
    output_dir="meta_results",
    label_col="subtype",
    id_col="patient_id",
)

results = run_inference(config)

# Print metrics
metrics = results["metrics"]["overall"]
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 (Macro): {metrics['f1_macro']:.3f}")
print(f"ROC AUC (Macro): {metrics['roc_auc']['macro']:.3f}")
```

### Example 3: Hierarchical Model with GPU

```python
config = InferenceConfig(
    run_dir="derived_hierarchical/fold1",
    data_csv="test_hierarchical.csv",
    output_dir="hierarchical_results",
    label_col="diagnosis",
    device="cuda",
    batch_size=1024,
)

results = run_inference(config)

# Access hierarchical predictions
preds = results["predictions"]
print(preds[["predicted_label_L1", "predicted_label_L2", "predicted_label"]])
```

### Example 4: Lenient Mode for Incomplete Data

```python
config = InferenceConfig(
    run_dir="derived/fold1",
    data_csv="incomplete_test.csv",
    output_dir="lenient_results",
    strict_features=False,
    lenient_fill_strategy="median",
    verbose=2,
)

results = run_inference(config)

# Check warnings
print(f"Warnings: {len(results['warnings'])}")
for w in results["warnings"]:
    print(f"  - {w}")
```

## Testing

Run the test suite:

```bash
# All inference tests
pytest tests/inference/ -v

# Specific test file
pytest tests/inference/test_preprocess.py -v

# With coverage
pytest tests/inference/ --cov=src/classiflow/inference --cov-report=html
```

## Troubleshooting

### Issue: "Missing required features"

**Solution:**
- Use lenient mode: `strict_features=False`
- Or ensure test data has all training features
- Check feature list: inspect `run_manifest.json` or training config

### Issue: "No valid samples for evaluation"

**Solution:**
- Ensure `label_col` contains valid labels
- Check for mismatched class names between training and test
- Verify labels are not all NaN

### Issue: "Probability matrix has wrong shape"

**Solution:**
- This indicates a class mismatch
- Ensure test labels match training classes
- Check `meta_classes.csv` in run directory

### Issue: "Model not found for hierarchical branch"

**Solution:**
- Some L1 classes may not have trained L2 models
- This is expected if insufficient data for that branch
- Predictions will fall back to L1 only

## Migration from Legacy Runs

For training runs without run manifests:

**Feature list:**
- Inferred from first pipeline
- Falls back to numeric columns
- Warning logged

**Best models:**
- Extracted from pipeline keys
- Uses last model per task

**Run type:**
- Detected from artifact filenames
- Checks for hierarchical, meta, or binary indicators

**Recommendation:** Re-run training with latest package version to generate full manifests.

## API Reference

### run_inference(config: InferenceConfig) → Dict[str, Any]

Main entry point for inference pipeline.

**Returns:**
- `predictions`: DataFrame with predictions
- `metrics`: Dict with computed metrics (if labels provided)
- `output_files`: Dict mapping file type to Path
- `warnings`: List of warning messages
- `config`: Dict of configuration used
- `timestamp`: ISO timestamp of run

### InferenceConfig

See [Configuration Options](#configuration-options) section above.

### ArtifactLoader

Low-level class for loading trained artifacts.

```python
from classiflow.inference.loader import ArtifactLoader

loader = ArtifactLoader("derived/fold1", fold=1)

# Load binary artifacts
pipes, best_models, features = loader.load_binary_artifacts(variant="smote")

# Load meta artifacts
meta_model, meta_features, meta_classes = loader.load_meta_artifacts(variant="smote")

# Get feature schema
schema = loader.get_feature_schema()
```

### FeatureAligner

Low-level class for feature preprocessing.

```python
from classiflow.inference.preprocess import FeatureAligner

aligner = FeatureAligner(
    required_features=["age", "size", "grade"],
    strict=False,
    fill_strategy="median",
)

X, metadata, warnings = aligner.align(df, id_col="patient_id", label_col="diagnosis")
```

## Contributing

When adding new features to the inference pipeline:

1. **Update config.py** if new configuration needed
2. **Add tests** in `tests/inference/`
3. **Update this README** with usage examples
4. **Ensure backwards compatibility** with existing run artifacts
5. **Document output formats** if new files generated

## License

Part of the classiflow package. See main package LICENSE.

## Citation

If you use this inference pipeline in your research, please cite:

```
[Citation information for classiflow package]
```

## Support

- **Issues**: https://github.com/your-org/classiflow/issues
- **Documentation**: See main package README
- **Examples**: See `examples/inference/` directory
