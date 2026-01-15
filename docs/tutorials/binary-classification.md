# Binary Classification Tutorial

This tutorial walks through a complete binary classification workflow using classiflow's nested cross-validation pipeline.

## What You'll Learn

- Preparing data for binary classification
- Configuring and running `train_binary_task`
- Understanding the nested CV structure
- Interpreting output metrics
- Running inference on new data

## Prerequisites

```bash
pip install classiflow[all]
```

## Step 1: Prepare Your Data

Classiflow expects a CSV file with:

- **Feature columns**: Numeric features for classification
- **Label column**: The target class (binary)
- **Optional**: ID column, patient/sample identifiers

For this tutorial, we'll create a dataset from scikit-learn's breast cancer data:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load built-in dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = ["Malignant" if t == 0 else "Benign" for t in data.target]
df["sample_id"] = [f"sample_{i:03d}" for i in range(len(df))]

# Save to CSV
output_dir = Path("outputs/01_binary_tutorial")
output_dir.mkdir(parents=True, exist_ok=True)
data_path = output_dir / "breast_cancer_data.csv"
df.to_csv(data_path, index=False)

print(f"Dataset saved to: {data_path}")
print(f"Shape: {df.shape}")
print(f"Class distribution:\n{df['diagnosis'].value_counts()}")
```

Expected output:
```
Dataset saved to: outputs/01_binary_tutorial/breast_cancer_data.csv
Shape: (569, 32)
Class distribution:
Benign       357
Malignant    212
Name: diagnosis, dtype: int64
```

## Step 2: Configure Training

Create a `TrainConfig` with your desired settings:

```python
from classiflow import TrainConfig

config = TrainConfig(
    # Data
    data_csv=data_path,
    label_col="diagnosis",
    pos_label="Malignant",  # Minority class as positive

    # Output
    outdir=output_dir / "run",

    # Cross-validation
    outer_folds=5,      # 5-fold outer CV for evaluation
    inner_splits=5,     # 5-fold inner CV for tuning
    inner_repeats=2,    # Repeat inner CV twice for stability
    random_state=42,    # For reproducibility

    # SMOTE (off for now, we'll compare later)
    smote_mode="off",
)

print("Configuration:")
for key, value in config.to_dict().items():
    print(f"  {key}: {value}")
```

!!! tip "Choosing outer_folds and inner_splits"
    - **outer_folds**: More folds = better estimate, but slower. 5-10 is typical.
    - **inner_splits**: Controls hyperparameter tuning granularity. 3-5 is common.
    - **inner_repeats**: Reduces variance in hyperparameter selection. 1-3 is typical.

## Step 3: Run Training

Execute the training pipeline:

```python
from classiflow import train_binary_task
import logging

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)

# Train
results = train_binary_task(config)

print("\nTraining complete!")
print(f"Run ID: {results.get('run_id', 'N/A')}")
```

This will:

1. Load and validate your data
2. Create train/test splits for each outer fold
3. For each outer fold:
   - Run inner CV with multiple models (LogisticRegression, SVM, RandomForest, etc.)
   - Select best hyperparameters per model
   - Evaluate on held-out test set
4. Aggregate metrics across folds
5. Save all artifacts

## Step 4: Understand the Output Structure

After training, examine the output directory:

```python
from pathlib import Path

def show_tree(path, prefix=""):
    """Display directory tree."""
    path = Path(path)
    contents = sorted(path.iterdir())
    for i, item in enumerate(contents):
        is_last = i == len(contents) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension)

show_tree(output_dir / "run")
```

Expected structure:
```
run/
├── run.json                 # Complete run manifest
├── config.json              # Training configuration
├── fold_1/
│   ├── binary_none/
│   │   ├── binary_pipes.joblib
│   │   ├── best_models.json
│   │   └── cv_results.csv
│   └── metrics_outer_eval.csv
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── summary_metrics.csv
```

## Step 5: Analyze Results

### Load Summary Metrics

```python
import pandas as pd

summary = pd.read_csv(output_dir / "run" / "summary_metrics.csv")
print("Summary Metrics (across all folds):")
print(summary.to_string(index=False))
```

Expected output:
```
Summary Metrics (across all folds):
    metric   mean    std    min    max
   roc_auc  0.994  0.004  0.988  0.999
  accuracy  0.968  0.015  0.947  0.991
        f1  0.958  0.022  0.929  0.987
 precision  0.953  0.027  0.915  0.988
    recall  0.967  0.024  0.929  1.000
```

### Load Per-Fold Metrics

```python
# Load metrics from each fold
fold_metrics = []
for fold_dir in sorted((output_dir / "run").glob("fold_*")):
    metrics_file = fold_dir / "metrics_outer_eval.csv"
    if metrics_file.exists():
        fold_df = pd.read_csv(metrics_file)
        fold_df["fold"] = fold_dir.name
        fold_metrics.append(fold_df)

all_folds = pd.concat(fold_metrics, ignore_index=True)
print("\nPer-fold metrics:")
print(all_folds.to_string(index=False))
```

### Examine Run Manifest

```python
import json

with open(output_dir / "run" / "run.json") as f:
    manifest = json.load(f)

print("\nRun Manifest:")
print(f"  Run ID: {manifest['run_id']}")
print(f"  Timestamp: {manifest['timestamp']}")
print(f"  Data hash: {manifest['training_data_hash'][:16]}...")
print(f"  Features: {len(manifest['feature_list'])} features")
print(f"  Git hash: {manifest.get('git_hash', 'N/A')}")
```

## Step 6: Run Inference

Apply your trained model to new data:

```python
from classiflow.inference import run_inference, InferenceConfig

# For demo, we'll use a subset of training data as "new" samples
test_df = df.sample(n=50, random_state=123)
test_path = output_dir / "test_samples.csv"
test_df.to_csv(test_path, index=False)

# Configure inference
infer_config = InferenceConfig(
    run_dir=output_dir / "run",
    data_csv=test_path,
    output_dir=output_dir / "inference",
    id_col="sample_id",
    label_col="diagnosis",  # Include for evaluation
)

# Run inference
infer_results = run_inference(infer_config)

print("\nInference complete!")
print(f"Predictions shape: {infer_results['predictions'].shape}")
```

### View Predictions

```python
predictions = infer_results["predictions"]
print("\nSample predictions:")
print(predictions[["sample_id", "diagnosis", "predicted_label", "binary_task_score"]].head(10))
```

### Check Inference Metrics

```python
if "metrics" in infer_results:
    metrics = infer_results["metrics"]
    print("\nInference metrics:")
    for key, value in metrics.get("overall", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
```

## Step 7: Visualize Results

Generate basic visualizations:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

# Get predictions
y_true = predictions["diagnosis"].map({"Malignant": 1, "Benign": 0})
y_score = predictions["binary_task_score"]
y_pred = predictions["predicted_label"].map({"Malignant": 1, "Benign": 0})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0])
axes[0].set_title("ROC Curve")

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=["Benign", "Malignant"],
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig(output_dir / "results_visualization.png", dpi=150)
plt.show()

print(f"\nVisualization saved to: {output_dir / 'results_visualization.png'}")
```

## Common Pitfalls

!!! warning "Data Leakage"
    Never include the label column in your features. Classiflow automatically excludes `label_col` from features, but be careful with derived features.

!!! warning "Small Sample Sizes"
    With fewer than 50 samples per class, nested CV may produce unstable estimates. Consider:

    - Reducing `outer_folds` to 3
    - Using stratified sampling
    - Reporting confidence intervals

!!! warning "Class Imbalance"
    If one class has <20% of samples, enable SMOTE:
    ```python
    config = TrainConfig(
        smote_mode="both",  # Compare with/without
        ...
    )
    ```
    See the [Imbalanced Data Tutorial](imbalanced-data.md) for details.

## What to Report in a Paper

When publishing results from this workflow, report:

1. **Dataset**: Sample size, feature count, class distribution
2. **Cross-validation**: Outer folds, inner splits/repeats
3. **Metrics**: Mean ± std across outer folds (at minimum: AUC, accuracy)
4. **Model selection**: Which model was selected most often
5. **Random seed**: For reproducibility

Example methods text:

> We performed binary classification using classiflow v0.1.0 with 5-fold nested cross-validation (5-fold inner CV with 2 repeats). The minority class (Malignant, n=212) was designated as positive. Model selection was performed using ROC-AUC as the primary metric. The final model achieved a mean AUC of 0.994 (SD=0.004) across outer folds.

## Next Steps

- [Multiclass Classification](multiclass-classification.md) - Handle 3+ classes
- [Imbalanced Data](imbalanced-data.md) - Enable SMOTE for rare classes
- [Publication Figures](publication-figures.md) - Generate manuscript-ready plots

## Complete Code

See `examples/01_quickstart_binary.py` for the complete runnable script.
