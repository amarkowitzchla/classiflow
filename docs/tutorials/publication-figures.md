# Publication Figures Tutorial

This tutorial shows how to generate publication-ready figures from classiflow results.

## What You'll Learn

- ROC curves with confidence intervals
- Precision-recall curves
- Confusion matrices
- Calibration plots
- Feature importance rankings
- Multi-panel figure layouts

## Prerequisites

```bash
pip install classiflow[all]
```

Ensure you have completed a training run:
```python
# Uses output from binary classification tutorial
run_dir = Path("outputs/01_binary_tutorial/run")
```

## Step 1: Load Results

```python
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, calibration_curve
)

output_dir = Path("outputs/05_publication_figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load a trained run
run_dir = Path("outputs/01_binary_tutorial/run")

# Load summary metrics
summary = pd.read_csv(run_dir / "summary_metrics.csv")
print("Available metrics:")
print(summary["metric"].tolist())
```

## Step 2: ROC Curves Across Folds

### Aggregate ROC with Confidence Band

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Collect ROC data from each fold
fig, ax = plt.subplots(figsize=(8, 8))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# For each fold, we need predictions (this is simplified - you'd load actual fold predictions)
# In practice, run inference on held-out data to get predictions

# Placeholder for illustration - replace with actual fold predictions
np.random.seed(42)
for fold in range(1, 6):
    # Simulate fold-level ROC (replace with actual predictions)
    y_true = np.random.binomial(1, 0.3, 100)
    y_score = y_true * 0.7 + np.random.random(100) * 0.3

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Interpolate to common FPR grid
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)

    # Plot individual fold (light)
    ax.plot(fpr, tpr, alpha=0.2, color="blue", linewidth=1)

# Mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color="blue", linewidth=2,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

# Confidence band (±1 std)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="blue", alpha=0.2,
                label="± 1 std")

# Reference line
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve with Cross-Validation Confidence Band", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(output_dir / "roc_curve_cv.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "roc_curve_cv.pdf", bbox_inches="tight")  # Vector format
plt.show()

print(f"ROC curve saved to: {output_dir / 'roc_curve_cv.png'}")
```

## Step 3: Precision-Recall Curve

PR curves are more informative for imbalanced datasets:

```python
fig, ax = plt.subplots(figsize=(8, 8))

# Collect PR data from each fold
precisions = []
mean_recall = np.linspace(0, 1, 100)
aps = []

for fold in range(1, 6):
    # Simulate (replace with actual)
    y_true = np.random.binomial(1, 0.3, 100)
    y_score = y_true * 0.7 + np.random.random(100) * 0.3

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    aps.append(ap)

    # Interpolate (note: PR curves go right to left)
    interp_precision = np.interp(mean_recall[::-1], recall[::-1], precision[::-1])
    precisions.append(interp_precision[::-1])

    ax.plot(recall, precision, alpha=0.2, color="green", linewidth=1)

# Mean PR curve
mean_precision = np.mean(precisions, axis=0)
mean_ap = np.mean(aps)
std_ap = np.std(aps)

ax.plot(mean_recall, mean_precision, color="green", linewidth=2,
        label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})")

# Confidence band
std_precision = np.std(precisions, axis=0)
ax.fill_between(mean_recall,
                np.maximum(mean_precision - std_precision, 0),
                np.minimum(mean_precision + std_precision, 1),
                color="green", alpha=0.2)

# Baseline (proportion of positives)
baseline = 0.3  # Replace with actual positive rate
ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.2f})")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve with Cross-Validation", fontsize=14)
ax.legend(loc="lower left", fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "pr_curve_cv.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Step 4: Confusion Matrix

```python
# Aggregate confusion matrix across folds
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example with real predictions (replace with actual)
y_true_all = np.concatenate([np.random.binomial(1, 0.37, 100) for _ in range(5)])
y_pred_all = (np.random.random(500) > 0.4).astype(int)

cm = confusion_matrix(y_true_all, y_pred_all)

fig, ax = plt.subplots(figsize=(7, 6))

# Normalized confusion matrix
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])

# Add percentages
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.7, f"({cm_normalized[i, j]:.1%})",
                ha="center", va="center", fontsize=10, color="gray")

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix (Aggregated Across Folds)", fontsize=14)

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Step 5: Calibration Plot

Calibration shows whether predicted probabilities match actual outcomes:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Calibration curve
ax1 = axes[0]

for fold in range(1, 6):
    y_true = np.random.binomial(1, 0.37, 200)
    y_prob = np.clip(y_true * 0.6 + np.random.random(200) * 0.4, 0, 1)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax1.plot(prob_pred, prob_true, marker="o", alpha=0.3, linewidth=1)

ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
ax1.set_ylabel("Fraction of Positives", fontsize=12)
ax1.set_title("Calibration Curve", fontsize=14)
ax1.legend()

# Right: Distribution of predictions
ax2 = axes[1]
y_prob_all = np.clip(np.random.random(1000) * 0.8 + 0.1, 0, 1)
ax2.hist(y_prob_all, bins=30, edgecolor="black", alpha=0.7)
ax2.set_xlabel("Predicted Probability", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Prediction Distribution", fontsize=14)

plt.tight_layout()
plt.savefig(output_dir / "calibration_plot.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Step 6: Feature Importance

```python
# Load feature importance from a trained model
# This requires loading the actual trained pipeline

# Example visualization (replace with actual importance values)
feature_names = [f"feature_{i}" for i in range(20)]
importances = np.abs(np.random.randn(20))
importances = importances / importances.sum()

# Sort by importance
idx = np.argsort(importances)[::-1][:15]  # Top 15

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(idx)))
ax.barh(range(len(idx)), importances[idx][::-1], color=colors[::-1])
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
ax.set_xlabel("Feature Importance (Normalized)", fontsize=12)
ax.set_title("Top 15 Features by Importance", fontsize=14)

plt.tight_layout()
plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Step 7: Multi-Panel Figure

Create a publication-ready multi-panel figure:

```python
fig = plt.figure(figsize=(14, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: ROC curve
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(mean_fpr, mean_tpr, color="blue", linewidth=2)
ax_a.fill_between(mean_fpr, tprs_lower, tprs_upper, color="blue", alpha=0.2)
ax_a.plot([0, 1], [0, 1], "k--", linewidth=1)
ax_a.set_xlabel("FPR")
ax_a.set_ylabel("TPR")
ax_a.set_title("A. ROC Curve")
ax_a.text(0.6, 0.2, f"AUC = {mean_auc:.3f}", fontsize=10)

# Panel B: PR curve
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(mean_recall, mean_precision, color="green", linewidth=2)
ax_b.set_xlabel("Recall")
ax_b.set_ylabel("Precision")
ax_b.set_title("B. Precision-Recall Curve")
ax_b.text(0.6, 0.8, f"AP = {mean_ap:.3f}", fontsize=10)

# Panel C: Confusion matrix
ax_c = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_c,
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
ax_c.set_xlabel("Predicted")
ax_c.set_ylabel("True")
ax_c.set_title("C. Confusion Matrix")

# Panel D: Metrics bar chart
ax_d = fig.add_subplot(gs[1, 0])
metrics = ["AUC", "Accuracy", "F1", "Recall", "Precision"]
values = [0.95, 0.92, 0.88, 0.90, 0.86]
errors = [0.02, 0.03, 0.04, 0.05, 0.04]
ax_d.bar(metrics, values, yerr=errors, capsize=3, color="steelblue", edgecolor="black")
ax_d.set_ylim(0, 1.1)
ax_d.set_ylabel("Score")
ax_d.set_title("D. Performance Metrics")

# Panel E: Calibration
ax_e = fig.add_subplot(gs[1, 1])
ax_e.plot([0, 1], [0, 1], "k--")
ax_e.plot([0.1, 0.3, 0.5, 0.7, 0.9], [0.12, 0.28, 0.51, 0.68, 0.88], "o-", color="purple")
ax_e.set_xlabel("Predicted Probability")
ax_e.set_ylabel("Observed Frequency")
ax_e.set_title("E. Calibration")

# Panel F: Feature importance
ax_f = fig.add_subplot(gs[1, 2])
top_n = 8
ax_f.barh(range(top_n), sorted(importances[:top_n], reverse=True), color="coral")
ax_f.set_yticks(range(top_n))
ax_f.set_yticklabels([f"Feature {i+1}" for i in range(top_n)])
ax_f.set_xlabel("Importance")
ax_f.set_title("F. Top Features")

plt.tight_layout()
plt.savefig(output_dir / "figure_combined.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "figure_combined.pdf", bbox_inches="tight")
plt.show()

print(f"\nMulti-panel figure saved to: {output_dir}")
```

## Figure Formatting Guidelines

### Journal Requirements

| Aspect | Typical Requirement |
|--------|---------------------|
| Resolution | 300 DPI minimum |
| Format | TIFF, PDF, or EPS for print |
| Font size | 8-12 pt in final size |
| Line width | ≥0.5 pt |
| Color | Consider colorblind-safe palettes |

### Colorblind-Safe Palettes

```python
# Use colorblind-friendly colors
cb_colors = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "cyan": "#56B4E9",
    "red": "#D55E00",
}
```

### Matplotlib Style for Publications

```python
# Publication-quality defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
})
```

## Output File Summary

After this tutorial, you should have:

```
outputs/05_publication_figures/
├── roc_curve_cv.png          # ROC with confidence band
├── roc_curve_cv.pdf          # Vector format
├── pr_curve_cv.png           # Precision-recall curve
├── confusion_matrix.png      # Confusion matrix heatmap
├── calibration_plot.png      # Calibration analysis
├── feature_importance.png    # Feature rankings
├── figure_combined.png       # Multi-panel figure
└── figure_combined.pdf       # Vector format
```

## Next Steps

- [Methods Text & Tables](methods-tables.md) - Generate text for manuscripts
- [Customize Plots Guide](../how-to/customize-plots.md) - Advanced customization
- [Publication Checklist](../publication/checklist.md) - Complete reporting guide
