# Customize Metrics

This guide covers adding custom scoring functions and evaluation metrics.

## Default Metrics

Classiflow computes these metrics by default:

| Metric | Description | When to Use |
|--------|-------------|-------------|
| `roc_auc` | Area under ROC curve | Ranking quality (default selection metric) |
| `accuracy` | Correct predictions / total | Balanced datasets |
| `balanced_accuracy` | Mean recall per class | Imbalanced datasets |
| `f1` | Harmonic mean of precision/recall | Balance precision and recall |
| `precision` | True positives / predicted positives | Minimize false positives |
| `recall` | True positives / actual positives | Minimize false negatives |

## Using Scikit-learn Scorers

Classiflow uses scikit-learn's scoring system:

```python
from sklearn.metrics import make_scorer, matthews_corrcoef, cohen_kappa_score

# Create custom scorers
mcc_scorer = make_scorer(matthews_corrcoef)
kappa_scorer = make_scorer(cohen_kappa_score)

# These can be passed to GridSearchCV
scoring = {
    "roc_auc": "roc_auc",
    "mcc": mcc_scorer,
    "kappa": kappa_scorer,
}
```

## Custom Metric Functions

### Define a Custom Metric

```python
from sklearn.metrics import confusion_matrix
import numpy as np

def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def youden_j_score(y_true, y_pred):
    """Youden's J statistic (sensitivity + specificity - 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity + specificity - 1

# Convert to scorer
from sklearn.metrics import make_scorer

specificity_scorer = make_scorer(specificity_score)
youden_scorer = make_scorer(youden_j_score)
```

### Probability-Based Metrics

```python
from sklearn.metrics import brier_score_loss, log_loss

def brier_score(y_true, y_prob):
    """Brier score (lower is better)."""
    return brier_score_loss(y_true, y_prob)

# Note: needs_proba=True for probability-based scorers
brier_scorer = make_scorer(
    brier_score,
    needs_proba=True,
    greater_is_better=False  # Lower is better
)

log_loss_scorer = make_scorer(
    log_loss,
    needs_proba=True,
    greater_is_better=False
)
```

## Compute Metrics Post-Training

### Load Predictions and Compute

```python
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    precision_recall_fscore_support
)

# Load predictions from inference
predictions = pd.read_csv("inference_results/predictions.csv")
y_true = predictions["diagnosis"].map({"Malignant": 1, "Benign": 0})
y_pred = predictions["predicted_label"].map({"Malignant": 1, "Benign": 0})
y_prob = predictions["binary_task_score"]

# Compute additional metrics
metrics = {
    "MCC": matthews_corrcoef(y_true, y_pred),
    "Kappa": cohen_kappa_score(y_true, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    "Specificity": specificity_score(y_true, y_pred),
    "Youden's J": youden_j_score(y_true, y_pred),
}

print("Additional Metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

### Per-Class Metrics

```python
# Precision, Recall, F1 per class
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1]
)

class_metrics = pd.DataFrame({
    "Class": ["Benign", "Malignant"],
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Support": support,
})
print("\nPer-Class Metrics:")
print(class_metrics.to_string(index=False))
```

## Threshold Optimization

### Find Optimal Threshold

```python
from sklearn.metrics import roc_curve, precision_recall_curve

# ROC-based (maximize Youden's J)
fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold_roc = thresholds_roc[optimal_idx]

print(f"Optimal threshold (Youden's J): {optimal_threshold_roc:.3f}")
print(f"  Sensitivity: {tpr[optimal_idx]:.3f}")
print(f"  Specificity: {1 - fpr[optimal_idx]:.3f}")

# PR-based (maximize F1)
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

print(f"\nOptimal threshold (F1): {optimal_threshold_pr:.3f}")
print(f"  Precision: {precision[optimal_idx_pr]:.3f}")
print(f"  Recall: {recall[optimal_idx_pr]:.3f}")
```

### Apply Custom Threshold

```python
# Default threshold is 0.5
y_pred_default = (y_prob >= 0.5).astype(int)

# Custom threshold
y_pred_custom = (y_prob >= optimal_threshold_roc).astype(int)

print(f"Default threshold (0.5):")
print(f"  Accuracy: {(y_true == y_pred_default).mean():.3f}")

print(f"Optimized threshold ({optimal_threshold_roc:.3f}):")
print(f"  Accuracy: {(y_true == y_pred_custom).mean():.3f}")
```

## Confidence Intervals

### Bootstrap CI for Metrics

```python
from scipy import stats

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        scores.append(metric_func(y_true[idx], y_pred[idx]))

    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper

from sklearn.metrics import accuracy_score, roc_auc_score

acc_mean, acc_low, acc_high = bootstrap_metric(
    y_true.values, y_pred.values, accuracy_score
)
print(f"Accuracy: {acc_mean:.3f} (95% CI: [{acc_low:.3f}, {acc_high:.3f}])")
```

## Multiclass Metrics

```python
from sklearn.metrics import classification_report

# Full classification report
report = classification_report(
    y_true_multiclass,
    y_pred_multiclass,
    target_names=["TypeA", "TypeB", "TypeC"],
    output_dict=True
)

# Convert to DataFrame
report_df = pd.DataFrame(report).T
print(report_df)
```

## Best Practices

!!! tip "Choose Metrics Based on Goals"
    - **Medical diagnosis**: High recall (catch all positives)
    - **Spam detection**: High precision (avoid false alarms)
    - **General balance**: F1 or MCC

!!! warning "Don't Optimize on Test Set"
    Threshold optimization should be done on validation data, not test data.

!!! note "Report Multiple Metrics"
    A single metric can be misleading. Report AUC, accuracy, and at least one class-aware metric (F1 or MCC).
