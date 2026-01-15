# Imbalanced Data & SMOTE Tutorial

This tutorial covers handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) applied correctly inside cross-validation folds.

## What You'll Learn

- Why class imbalance matters
- How SMOTE works and its limitations
- Correct vs. incorrect SMOTE application
- Using `smote_mode` in classiflow
- Comparing results with and without SMOTE

## The Class Imbalance Problem

Class imbalance occurs when one class has significantly fewer samples than others. This causes:

- **Biased models** that favor the majority class
- **Misleading accuracy** (predicting all majority = high accuracy)
- **Poor minority class performance** (what you often care about most)

```python
# Example: 90% Class A, 10% Class B
# A model predicting "Class A" always achieves 90% accuracy
# but fails completely on the important minority class
```

## SMOTE: Synthetic Minority Over-sampling

SMOTE creates synthetic samples by interpolating between existing minority class samples:

1. For each minority sample, find k nearest neighbors
2. Create new samples along lines connecting to neighbors
3. Result: More minority samples, better class balance

!!! warning "SMOTE Must Be Applied Inside CV Folds"
    Applying SMOTE before splitting creates **data leakage**: synthetic samples in the test set may be based on training samples. This leads to overly optimistic performance estimates.

    **Correct**: SMOTE inside each training fold only
    **Incorrect**: SMOTE on full dataset before CV

Classiflow handles this correctly by default.

## Step 1: Create Imbalanced Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path

# Create imbalanced dataset (90/10 split)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    flip_y=0.01,
    random_state=42,
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
df["diagnosis"] = ["Common" if label == 0 else "Rare" for label in y]
df["sample_id"] = [f"sample_{i:04d}" for i in range(len(df))]

output_dir = Path("outputs/03_imbalanced_tutorial")
output_dir.mkdir(parents=True, exist_ok=True)
data_path = output_dir / "imbalanced_data.csv"
df.to_csv(data_path, index=False)

print("Class distribution:")
print(df["diagnosis"].value_counts())
print(f"\nImbalance ratio: {df['diagnosis'].value_counts()['Common'] / df['diagnosis'].value_counts()['Rare']:.1f}:1")
```

Output:
```
Class distribution:
Common    896
Rare      104
Name: diagnosis, dtype: int64

Imbalance ratio: 8.6:1
```

## Step 2: Train Without SMOTE (Baseline)

```python
from classiflow import train_binary_task, TrainConfig
import logging

logging.basicConfig(level=logging.INFO)

config_baseline = TrainConfig(
    data_csv=data_path,
    label_col="diagnosis",
    pos_label="Rare",  # Minority as positive
    smote_mode="off",
    outer_folds=5,
    inner_splits=3,
    random_state=42,
    outdir=output_dir / "run_baseline",
)

results_baseline = train_binary_task(config_baseline)
print("\nBaseline training complete!")
```

## Step 3: Train With SMOTE

```python
config_smote = TrainConfig(
    data_csv=data_path,
    label_col="diagnosis",
    pos_label="Rare",
    smote_mode="on",  # Enable SMOTE
    smote_k_neighbors=5,  # K for k-NN in SMOTE
    outer_folds=5,
    inner_splits=3,
    random_state=42,
    outdir=output_dir / "run_smote",
)

results_smote = train_binary_task(config_smote)
print("\nSMOTE training complete!")
```

## Step 4: Train Both for Comparison

The `smote_mode="both"` option trains with and without SMOTE in parallel:

```python
config_both = TrainConfig(
    data_csv=data_path,
    label_col="diagnosis",
    pos_label="Rare",
    smote_mode="both",  # Train both variants
    smote_k_neighbors=5,
    outer_folds=5,
    inner_splits=3,
    random_state=42,
    outdir=output_dir / "run_comparison",
)

results_both = train_binary_task(config_both)
print("\nComparison training complete!")
```

This creates two subdirectories per fold:
- `fold_X/binary_none/` - Without SMOTE
- `fold_X/binary_smote/` - With SMOTE

## Step 5: Compare Results

```python
import pandas as pd

# Load metrics from both runs
def load_summary(run_dir):
    summary_file = run_dir / "summary_metrics.csv"
    if summary_file.exists():
        return pd.read_csv(summary_file)
    return None

summary_baseline = load_summary(output_dir / "run_baseline")
summary_smote = load_summary(output_dir / "run_smote")

# Compare key metrics
metrics_to_compare = ["roc_auc", "f1", "recall", "precision", "balanced_accuracy"]

print("=" * 60)
print("COMPARISON: Baseline vs SMOTE")
print("=" * 60)
print(f"{'Metric':<20} {'Baseline':<15} {'SMOTE':<15} {'Change':<10}")
print("-" * 60)

for metric in metrics_to_compare:
    baseline_row = summary_baseline[summary_baseline["metric"] == metric]
    smote_row = summary_smote[summary_smote["metric"] == metric]

    if not baseline_row.empty and not smote_row.empty:
        baseline_val = baseline_row["mean"].values[0]
        smote_val = smote_row["mean"].values[0]
        change = smote_val - baseline_val

        print(f"{metric:<20} {baseline_val:.3f} ± {baseline_row['std'].values[0]:.3f}  "
              f"{smote_val:.3f} ± {smote_row['std'].values[0]:.3f}  "
              f"{change:+.3f}")
```

Expected output (varies by data):
```
============================================================
COMPARISON: Baseline vs SMOTE
============================================================
Metric               Baseline        SMOTE           Change
------------------------------------------------------------
roc_auc              0.923 ± 0.031   0.941 ± 0.025   +0.018
f1                   0.612 ± 0.089   0.687 ± 0.072   +0.075
recall               0.548 ± 0.112   0.721 ± 0.098   +0.173
precision            0.698 ± 0.095   0.659 ± 0.087   -0.039
balanced_accuracy    0.759 ± 0.054   0.831 ± 0.048   +0.072
```

!!! note "Interpreting the Comparison"
    - **Recall improved**: SMOTE helps detect more true positives
    - **Precision decreased slightly**: Some false positives introduced
    - **F1 improved**: Overall better balance
    - **Balanced accuracy improved**: Better performance on both classes

## Step 6: Visualize the Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

# Comparison bar chart
metrics = ["roc_auc", "f1", "recall", "precision", "balanced_accuracy"]
baseline_means = []
baseline_stds = []
smote_means = []
smote_stds = []

for metric in metrics:
    b_row = summary_baseline[summary_baseline["metric"] == metric]
    s_row = summary_smote[summary_smote["metric"] == metric]
    baseline_means.append(b_row["mean"].values[0] if not b_row.empty else 0)
    baseline_stds.append(b_row["std"].values[0] if not b_row.empty else 0)
    smote_means.append(s_row["mean"].values[0] if not s_row.empty else 0)
    smote_stds.append(s_row["std"].values[0] if not s_row.empty else 0)

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, baseline_means, width, label="Baseline (no SMOTE)",
               yerr=baseline_stds, capsize=3, color="#1f77b4")
bars2 = ax.bar(x + width/2, smote_means, width, label="With SMOTE",
               yerr=smote_stds, capsize=3, color="#2ca02c")

ax.set_ylabel("Score")
ax.set_title("Performance Comparison: Baseline vs SMOTE")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "smote_comparison.png", dpi=150)
plt.show()
```

## SMOTE Parameters

### smote_k_neighbors

Controls how many neighbors SMOTE uses for interpolation:

```python
# More neighbors = smoother interpolation, but may miss clusters
config = TrainConfig(
    smote_k_neighbors=3,  # For very small minority classes
    # ...
)

config = TrainConfig(
    smote_k_neighbors=7,  # For larger minority classes
    # ...
)
```

!!! tip "Choosing k_neighbors"
    - **Small minority class (<50 samples)**: Use k=3-5
    - **Medium minority class (50-200)**: Use k=5-7
    - **Large minority class (>200)**: Use k=5-10 or don't use SMOTE

### Adaptive SMOTE

Classiflow's `AdaptiveSMOTE` automatically adjusts k based on minority class size:

```python
from classiflow.models import AdaptiveSMOTE

# Automatically sets k_neighbors based on minority count
smote = AdaptiveSMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

## When NOT to Use SMOTE

!!! warning "SMOTE Limitations"

    **Don't use SMOTE when:**

    1. **Minority class is noisy** - SMOTE will amplify noise
    2. **Minority samples are outliers** - Synthetic samples will be unrealistic
    3. **Classes overlap significantly** - SMOTE may create ambiguous samples
    4. **Minority class is extremely small** (<10 samples) - Not enough for interpolation

    **Alternatives:**

    - **Class weights**: Penalize misclassification of minority class
    - **Threshold adjustment**: Lower classification threshold for minority
    - **Cost-sensitive learning**: Asymmetric loss functions

## Class Weights Alternative

Classiflow models support class weights (automatically handled by scikit-learn):

```python
# Class weights are applied internally when SMOTE is off
# For manual control, use sklearn estimators directly
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight="balanced")
```

## Comparing SMOTE vs Class Weights

```python
# Coming soon: classiflow will support class_weight_mode
# For now, use smote_mode and compare with external class weight runs
```

## What to Report

When publishing results with SMOTE:

1. **Class distribution** before and after SMOTE
2. **SMOTE parameters** (k_neighbors)
3. **Comparison** with/without SMOTE
4. **Note that SMOTE was applied inside CV folds**

Example methods text:

> Due to class imbalance (Common: n=896, Rare: n=104, ratio 8.6:1), we applied SMOTE with k=5 nearest neighbors. SMOTE was applied only to training folds to prevent data leakage. Models trained with SMOTE achieved higher balanced accuracy (0.831 vs. 0.759) and recall (0.721 vs. 0.548) compared to baseline, indicating improved minority class detection.

## Common Pitfalls

!!! warning "Leakage via Pre-Split SMOTE"
    ```python
    # WRONG: SMOTE before splitting
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

    # CORRECT: SMOTE inside each fold (classiflow does this automatically)
    ```

!!! warning "Overfitting with Aggressive SMOTE"
    Creating too many synthetic samples can lead to overfitting. Classiflow balances classes to 1:1 by default.

!!! warning "Ignoring Precision Drop"
    SMOTE often increases recall but decreases precision. Evaluate based on your application's needs.

## Next Steps

- [Cross-Validation Tutorial](cross-validation.md) - Understand nested CV
- [Metrics Interpretation](../concepts/metrics-interpretation.md) - Choose the right metrics
- [SMOTE Comparison Guide](../how-to/class-imbalance.md) - Advanced imbalance handling
