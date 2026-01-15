# Quickstart

Get up and running with classiflow in under 5 minutes.

## Minimal Binary Classification

Here's the simplest possible workflow—train a binary classifier with nested cross-validation:

```python
from classiflow import train_binary_task, TrainConfig

# Configure the training run
config = TrainConfig(
    data_csv="data/features.csv",
    label_col="diagnosis",
    pos_label="Malignant",
    outer_folds=5,
    outdir="derived/quickstart",
)

# Train with nested CV
results = train_binary_task(config)

# Check results
print(f"Mean AUC: {results['summary']['roc_auc']['mean']:.3f}")
```

That's it! Classiflow handles nested cross-validation, model selection, and metric computation automatically.

## Using the CLI

Prefer the command line? The same workflow via CLI:

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Malignant \
  --outer-folds 5 \
  --outdir derived/quickstart
```

## What Gets Generated

After training, your output directory contains:

```
derived/quickstart/
├── run.json                    # Complete run manifest (lineage)
├── config.json                 # Training configuration
├── fold_1/
│   ├── binary_none/
│   │   ├── binary_pipes.joblib # Trained pipelines
│   │   ├── best_models.json    # Best model per metric
│   │   └── cv_results.csv      # Cross-validation results
│   └── metrics_outer_eval.csv  # Outer fold metrics
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── summary_metrics.csv         # Aggregated metrics across folds
```

## Understanding the Output

### Run Manifest (`run.json`)

Contains complete lineage for reproducibility:

```json
{
  "run_id": "a1b2c3d4-...",
  "timestamp": "2024-01-15T10:30:00",
  "training_data_hash": "sha256:abc123...",
  "config": { ... },
  "feature_list": ["feature_1", "feature_2", ...],
  "best_models": {"binary_task": "LogisticRegression"}
}
```

### Summary Metrics

The `summary_metrics.csv` aggregates performance across outer folds:

| metric | mean | std | min | max |
|--------|------|-----|-----|-----|
| roc_auc | 0.892 | 0.023 | 0.861 | 0.918 |
| accuracy | 0.847 | 0.031 | 0.812 | 0.883 |
| f1 | 0.823 | 0.029 | 0.789 | 0.856 |

## Running Inference

Apply your trained model to new data:

```python
from classiflow.inference import run_inference, InferenceConfig

config = InferenceConfig(
    run_dir="derived/quickstart",
    data_csv="data/new_samples.csv",
    output_dir="derived/inference_results",
    label_col="diagnosis",  # Optional: include for metrics
)

results = run_inference(config)

# Access predictions
predictions = results["predictions"]
print(predictions[["sample_id", "predicted_label", "predicted_proba"]].head())
```

Or via CLI:

```bash
classiflow infer \
  --run-dir derived/quickstart \
  --data-csv data/new_samples.csv \
  --outdir derived/inference_results \
  --label-col diagnosis
```

## Enabling SMOTE

For imbalanced datasets, enable SMOTE (applied correctly inside CV folds):

```python
config = TrainConfig(
    data_csv="data/features.csv",
    label_col="diagnosis",
    pos_label="Rare_Subtype",
    smote_mode="both",  # Train with and without SMOTE
    smote_k_neighbors=5,
    outdir="derived/smote_comparison",
)
```

This trains two variants and lets you compare performance.

## Multiclass Classification

For more than two classes, use the meta-classifier:

```python
from classiflow import train_meta_classifier, MetaConfig

config = MetaConfig(
    data_csv="data/features.csv",
    label_col="subtype",
    classes=["TypeA", "TypeB", "TypeC"],
    outer_folds=5,
    outdir="derived/multiclass",
)

results = train_meta_classifier(config)
```

Or via CLI:

```bash
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outer-folds 5 \
  --outdir derived/multiclass
```

## Hierarchical Classification

For two-level classification (e.g., tumor type → subtype):

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --device auto \
  --outdir derived/hierarchical
```

## Statistical Analysis

Run publication-ready statistical comparisons:

```bash
classiflow stats run \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --outdir derived/stats

classiflow stats viz \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --stats-dir derived/stats \
  --outdir derived/stats/plots
```

This generates:

- Pairwise statistical tests (t-test, Mann-Whitney)
- Effect sizes (Cohen's d, log2 fold change)
- Volcano plots, heatmaps, boxplots
- Publication-ready Excel workbook

## Creating Portable Bundles

Package your model for sharing:

```bash
classiflow bundle create \
  --run-dir derived/quickstart \
  --out model_bundle.zip \
  --description "Binary classifier v1.0"
```

Inspect a bundle:

```bash
classiflow bundle inspect model_bundle.zip
```

## Next Steps

Now that you have the basics:

- [Binary Classification Tutorial](../tutorials/binary-classification.md) - Deep dive into binary workflows
- [Multiclass Tutorial](../tutorials/multiclass-classification.md) - OvR and pairwise ensembles
- [SMOTE & Imbalanced Data](../tutorials/imbalanced-data.md) - Handling class imbalance correctly
- [Publication Figures](../tutorials/publication-figures.md) - Generate manuscript-ready outputs
- [CLI Reference](../cli/index.md) - Complete command documentation
