# Cross-Validation & Reproducibility Tutorial

This tutorial explains classiflow's nested cross-validation approach and how to ensure reproducible results.

## What You'll Learn

- Why nested CV is essential for unbiased evaluation
- How classiflow's nested CV works
- Setting random seeds for reproducibility
- Interpreting variance across folds
- Capturing the full environment

## Why Nested Cross-Validation?

Standard k-fold CV has a critical flaw: **hyperparameter selection and performance evaluation use the same data**, leading to optimistic bias.

### The Problem with Standard CV

```python
# PROBLEMATIC: Single-level CV
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
best_score = grid_search.best_score_  # ← This is biased!
```

The `best_score_` is optimistically biased because:
1. We searched for hyperparameters that maximize this score
2. We're reporting the same score as our final estimate

### The Solution: Nested CV

```
┌─────────────────────────────────────────────────────────────────┐
│ OUTER LOOP: Performance Estimation                              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Fold 1: Train on folds 2-5, Test on fold 1              │   │
│   │                                                         │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │ INNER LOOP: Hyperparameter Tuning               │   │   │
│   │   │ GridSearchCV on training data (folds 2-5)       │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                         │   │
│   │   → Best model evaluated on held-out fold 1             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Repeat for folds 2, 3, 4, 5...                               │
│                                                                 │
│   Final estimate = mean(fold scores) ± std                      │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Configure Nested CV

```python
from classiflow import TrainConfig
from pathlib import Path

output_dir = Path("outputs/04_cv_tutorial")
output_dir.mkdir(parents=True, exist_ok=True)

config = TrainConfig(
    data_csv="outputs/01_binary_tutorial/breast_cancer_data.csv",
    label_col="diagnosis",
    pos_label="Malignant",

    # Outer loop: for unbiased performance estimation
    outer_folds=5,  # 5-fold outer CV

    # Inner loop: for hyperparameter tuning
    inner_splits=5,  # 5-fold inner CV
    inner_repeats=2,  # Repeat inner CV twice

    # Reproducibility
    random_state=42,

    outdir=output_dir / "run",
)
```

### Parameter Guide

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `outer_folds` | Unbiased performance estimation | 5-10 |
| `inner_splits` | Hyperparameter tuning granularity | 3-5 |
| `inner_repeats` | Reduce tuning variance | 1-3 |
| `random_state` | Reproducibility seed | Any integer |

!!! tip "Balancing Computation"
    Total CV iterations = `outer_folds × inner_splits × inner_repeats × n_hyperparameter_combinations`

    For large datasets, consider reducing `inner_repeats` to 1.

## Step 2: Understanding Random Seeds

### What `random_state` Controls

```python
config = TrainConfig(
    random_state=42,  # Controls:
    # 1. Outer fold stratified splitting
    # 2. Inner fold stratified splitting
    # 3. Model random states (where applicable)
    # 4. SMOTE random sampling (if enabled)
)
```

### Verifying Reproducibility

```python
from classiflow import train_binary_task
import pandas as pd

# Run 1
results1 = train_binary_task(config)

# Run 2 (same seed)
config2 = TrainConfig(**config.to_dict())
config2.outdir = output_dir / "run_verify"
results2 = train_binary_task(config2)

# Compare metrics
summary1 = pd.read_csv(output_dir / "run" / "summary_metrics.csv")
summary2 = pd.read_csv(output_dir / "run_verify" / "summary_metrics.csv")

print("Metrics match:", summary1.equals(summary2))
```

!!! note "Sources of Non-Determinism"
    Even with fixed seeds, slight variations may occur due to:

    - **Floating-point operations**: GPU vs. CPU, different hardware
    - **Library versions**: scikit-learn, numpy updates
    - **Parallel execution**: Thread ordering

    Classiflow minimizes these but cannot eliminate all sources.

## Step 3: Examining Fold-Level Results

```python
import pandas as pd
from pathlib import Path

# Load per-fold metrics
fold_metrics = []
run_dir = output_dir / "run"

for fold_dir in sorted(run_dir.glob("fold_*")):
    metrics_file = fold_dir / "metrics_outer_eval.csv"
    if metrics_file.exists():
        fold_df = pd.read_csv(metrics_file)
        fold_df["fold"] = int(fold_dir.name.split("_")[1])
        fold_metrics.append(fold_df)

all_folds = pd.concat(fold_metrics, ignore_index=True)

print("Per-fold results:")
print(all_folds.pivot(index="fold", columns="metric", values="value"))
```

### Interpreting Variance

```python
# Calculate variance statistics
for metric in ["roc_auc", "accuracy", "f1"]:
    metric_values = all_folds[all_folds["metric"] == metric]["value"]
    print(f"\n{metric}:")
    print(f"  Mean: {metric_values.mean():.4f}")
    print(f"  Std:  {metric_values.std():.4f}")
    print(f"  Range: [{metric_values.min():.4f}, {metric_values.max():.4f}]")
```

!!! warning "High Variance Across Folds"
    If std > 0.05 for AUC or accuracy, consider:

    1. **More outer folds**: Better estimate of variance
    2. **More samples**: Small datasets have high variance
    3. **Stratification issues**: Check class balance per fold

## Step 4: Environment Capture

### Automatic Capture via Run Manifest

Classiflow automatically records:

```python
import json

with open(output_dir / "run" / "run.json") as f:
    manifest = json.load(f)

print("Environment captured:")
print(f"  Package version: {manifest.get('package_version', 'N/A')}")
print(f"  Python version:  {manifest.get('python_version', 'N/A')}")
print(f"  Git hash:        {manifest.get('git_hash', 'N/A')[:8]}...")
print(f"  Hostname:        {manifest.get('hostname', 'N/A')}")
print(f"  Timestamp:       {manifest.get('timestamp', 'N/A')}")
```

### Manual Environment Export

For complete reproducibility, export your environment:

```bash
# Pip
pip freeze > requirements_frozen.txt

# Conda
conda env export > environment.yml

# Include in your project
```

### Data Hashing

The run manifest includes data hash for verification:

```python
print(f"Training data hash: {manifest['training_data_hash']}")
print(f"Data row count:     {manifest['training_data_row_count']}")
```

## Step 5: Reporting Reproducibility

### Minimum Information to Report

1. **Software versions**: classiflow, Python, scikit-learn
2. **CV structure**: outer_folds, inner_splits, inner_repeats
3. **Random seed**: For exact reproducibility
4. **Hardware**: CPU/GPU, memory (if relevant)

### Example Methods Section

> Classification was performed using classiflow v0.1.0 (Python 3.10, scikit-learn 1.4.0). We used 5-fold nested cross-validation with 5-fold inner CV repeated twice for hyperparameter tuning. All random states were fixed to 42 for reproducibility. The complete run manifest and frozen requirements are available in the supplementary materials.

### Reproducibility Checklist

- [ ] Random seed fixed and reported
- [ ] CV structure documented
- [ ] Software versions recorded
- [ ] Data hash captured
- [ ] Environment file exported
- [ ] Git commit hash recorded

## Step 6: Handling Non-Reproducibility

### Debugging Differences

```python
import classiflow
import sklearn
import numpy as np
import pandas as pd

print("Version check:")
print(f"  classiflow: {classiflow.__version__}")
print(f"  sklearn:    {sklearn.__version__}")
print(f"  numpy:      {np.__version__}")
print(f"  pandas:     {pd.__version__}")
```

### Common Causes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Different metrics each run | Random seed not fixed | Set `random_state` |
| Different results on different machines | Library versions | Use exact versions |
| GPU vs CPU differences | Floating-point order | Use deterministic mode |
| Slight metric variations | Parallel processing | Set `n_jobs=1` |

## Confidence Intervals

### Bootstrap Confidence Intervals

```python
import numpy as np
from scipy import stats

def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper

# Example: AUC confidence interval
auc_values = all_folds[all_folds["metric"] == "roc_auc"]["value"].values
ci_low, ci_high = bootstrap_ci(auc_values)
print(f"AUC 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
```

### Reporting with CI

> The model achieved a mean AUC of 0.934 (95% CI: 0.912-0.956) across 5-fold nested cross-validation.

## Best Practices Summary

1. **Always use nested CV** for unbiased performance estimates
2. **Fix random seeds** for reproducibility
3. **Report variance** (std or CI) alongside mean
4. **Capture environment** (versions, hardware)
5. **Store run manifests** with results
6. **Use version control** for code and configs

## Next Steps

- [Publication Figures](publication-figures.md) - Generate manuscript-ready visualizations
- [Concepts: Reproducibility](../concepts/reproducibility.md) - Deep dive into determinism
- [API: TrainConfig](../api/config.md) - Complete configuration reference
