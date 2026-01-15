# Handle Class Imbalance

This guide covers strategies for handling imbalanced datasets responsibly.

## Identifying Imbalance

```python
import pandas as pd

df = pd.read_csv("data/features.csv")
class_counts = df["diagnosis"].value_counts()
print("Class distribution:")
print(class_counts)

# Calculate imbalance ratio
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")

# Severity guidelines
if imbalance_ratio < 2:
    print("Mild imbalance - standard methods likely sufficient")
elif imbalance_ratio < 5:
    print("Moderate imbalance - consider SMOTE or class weights")
else:
    print("Severe imbalance - use multiple strategies")
```

## Strategy 1: SMOTE

SMOTE creates synthetic minority samples by interpolating between neighbors.

### When to Use SMOTE

- Moderate imbalance (2:1 to 10:1)
- Sufficient minority samples (>30)
- Features are continuous

### Configuration

```python
from classiflow import TrainConfig

config = TrainConfig(
    data_csv="data/features.csv",
    label_col="diagnosis",
    pos_label="Rare",

    # Enable SMOTE
    smote_mode="both",  # Compare with and without
    smote_k_neighbors=5,  # Adjust based on minority size

    outdir="derived/run_smote",
)
```

### Choosing `smote_k_neighbors`

| Minority Size | Recommended k |
|---------------|---------------|
| <20 | 3 |
| 20-50 | 5 |
| 50-100 | 5-7 |
| >100 | 5-10 |

!!! warning "SMOTE Must Be Inside CV"
    Classiflow applies SMOTE correctly—inside each training fold only. This prevents data leakage.

## Strategy 2: Class Weights

Penalize misclassification of minority class more heavily.

### Advantages Over SMOTE

- No synthetic data created
- Works with any sample size
- Computationally cheaper

### Using Class Weights

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute balanced weights
classes = np.unique(y)
weights = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# Use in model
model = LogisticRegression(class_weight="balanced")
# or
model = LogisticRegression(class_weight=class_weights)
```

!!! note "Classiflow Support"
    Class weights are applied automatically when using `smote_mode="off"` with models that support them.

## Strategy 3: Threshold Tuning

Adjust classification threshold to favor minority class.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Find threshold that maximizes F1
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Precision at threshold: {precision[optimal_idx]:.3f}")
print(f"Recall at threshold: {recall[optimal_idx]:.3f}")

# Apply threshold
y_pred_tuned = (y_prob >= optimal_threshold).astype(int)
```

!!! warning "Tune on Validation Set"
    Never tune threshold on test data. Use a held-out validation set or inner CV.

## Strategy 4: Ensemble Methods

### Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy="all",  # Undersample majority
    random_state=42,
)
```

### EasyEnsemble

```python
from imblearn.ensemble import EasyEnsembleClassifier

model = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42,
)
```

## Strategy 5: Cost-Sensitive Learning

Define asymmetric costs for different types of errors.

```python
# Example: False negatives cost 5x more than false positives
sample_weights = np.where(y == 1, 5.0, 1.0)

model.fit(X, y, sample_weight=sample_weights)
```

## Choosing Metrics for Imbalanced Data

| Metric | Use When |
|--------|----------|
| **AUPRC** | Minority class is important |
| **Balanced Accuracy** | Overall performance on both classes |
| **F1 Score** | Balance precision and recall |
| **MCC** | Comprehensive single number |

**Avoid**:
- **Accuracy**: Misleading when imbalanced
- **AUC-ROC alone**: Can hide poor minority performance

## Comparison Framework

```python
from classiflow import train_binary_task, TrainConfig

# Run 1: No resampling
config_baseline = TrainConfig(
    smote_mode="off",
    outdir="derived/compare/baseline",
    # ... other params
)

# Run 2: SMOTE
config_smote = TrainConfig(
    smote_mode="on",
    smote_k_neighbors=5,
    outdir="derived/compare/smote",
    # ... other params
)

# Train both
results_baseline = train_binary_task(config_baseline)
results_smote = train_binary_task(config_smote)

# Compare on minority-class metrics
print("Baseline recall:", results_baseline["summary"]["recall"]["mean"])
print("SMOTE recall:", results_smote["summary"]["recall"]["mean"])
```

## Recommendations by Imbalance Level

### Mild (2:1 - 3:1)
- Standard methods often work
- Use balanced accuracy as primary metric
- Consider class weights

### Moderate (3:1 - 10:1)
- Use `smote_mode="both"` to compare
- Report AUPRC alongside AUC-ROC
- Consider threshold tuning

### Severe (>10:1)
- Combine strategies: SMOTE + class weights
- Focus on AUPRC, F1, MCC
- Consider ensemble methods
- Report per-class metrics

## What to Report

When publishing results on imbalanced data:

1. **Class distribution** before any resampling
2. **Imbalance handling strategy** used
3. **Metrics appropriate for imbalance** (AUPRC, balanced accuracy)
4. **Per-class metrics** (not just overall)
5. **Comparison** of different strategies if applicable

Example:

> Due to class imbalance (Rare: n=45, Common: n=455, ratio 10:1), we applied SMOTE (k=5) within training folds. Models were evaluated using AUPRC (0.72) and balanced accuracy (0.78) as primary metrics. SMOTE improved recall from 0.62 to 0.79 with minimal precision loss (0.71 to 0.68).
