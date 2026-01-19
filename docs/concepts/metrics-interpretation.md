# Metrics Interpretation

Understanding classification metrics, especially for imbalanced datasets.

## The Confusion Matrix

All metrics derive from the confusion matrix:

```
                    Predicted
                Negative    Positive
Actual  Negative    TN          FP
        Positive    FN          TP
```

- **TP**: True Positives (correctly identified positives)
- **TN**: True Negatives (correctly identified negatives)
- **FP**: False Positives (Type I error)
- **FN**: False Negatives (Type II error)

## Basic Metrics

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Interpretation**: Fraction of correct predictions
- **Use when**: Classes are balanced
- **Avoid when**: Imbalanced (can be misleading)

!!! warning "Accuracy Paradox"
    With 95% majority class, predicting all majority achieves 95% accuracy but 0% minority detection.

### Precision (Positive Predictive Value)

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Interpretation**: Of predicted positives, how many are correct?
- **Optimize when**: False positives are costly (e.g., spam detection)

### Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **Interpretation**: Of actual positives, how many were detected?
- **Optimize when**: False negatives are costly (e.g., disease screening)

### Specificity (True Negative Rate)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

- **Interpretation**: Of actual negatives, how many were correctly identified?

### F1 Score

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Interpretation**: Harmonic mean of precision and recall
- **Use when**: You need balance between precision and recall
- **Range**: 0 to 1 (higher is better)

## Threshold-Independent Metrics

### AUC-ROC (Area Under ROC Curve)

The ROC curve plots TPR vs FPR at various thresholds.

- **Interpretation**: Probability that a random positive ranks higher than a random negative
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Use when**: Overall ranking quality matters

!!! note "AUC Interpretation Guidelines"
    | AUC | Quality |
    |-----|---------|
    | 0.9-1.0 | Excellent |
    | 0.8-0.9 | Good |
    | 0.7-0.8 | Fair |
    | 0.6-0.7 | Poor |
    | 0.5-0.6 | Failed |

### AUC-PRC (Area Under Precision-Recall Curve)

The PR curve plots precision vs recall at various thresholds.

- **Interpretation**: Summarizes precision-recall trade-off
- **Range**: Baseline (minority proportion) to 1.0
- **Use when**: Imbalanced data, minority class is important

!!! tip "AUC-PRC for Imbalanced Data"
    AUC-ROC can be overly optimistic with severe imbalance. AUC-PRC is more informative for minority class performance.

## Balanced Metrics

### Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

- **Interpretation**: Average of recall per class
- **Use when**: Imbalanced data, equal importance per class

### Matthews Correlation Coefficient (MCC)

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

- **Interpretation**: Correlation between predicted and actual
- **Range**: -1 (total disagreement) to +1 (perfect)
- **Use when**: You want a single comprehensive metric
- **Advantage**: Accounts for all four quadrants of confusion matrix

!!! tip "MCC is Robust"
    MCC produces a high score only if the classifier does well on all four confusion matrix quadrants, making it resistant to class imbalance.

## Calibration

### What is Calibration?

A model is **well-calibrated** if predicted probabilities match observed frequencies:

- If you predict P=0.8 for 100 samples, about 80 should be positive

### Brier Score

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

- **Interpretation**: Mean squared error of probability estimates
- **Range**: 0 (perfect) to 1 (worst)

### Expected Calibration Error (ECE)

Groups predictions into bins and measures calibration per bin.

- **Use when**: You need well-calibrated probabilities (e.g., risk scores)

### Why Calibration Gates Promotion

High AUC can still produce misleading probabilities, so ROC AUC alone is not enough
for clinical deployment. Clinical workflows often rely on probabilities for triage,
counseling, and downstream decision support. Calibration metrics such as Brier and ECE
ensure probability quality, not just ranking.

In classiflow, promotion gates use calibrated Brier and ECE to confirm that a model
is both discriminative and probabilistically reliable.

- **Brier**: squared error of predicted probabilities; values closer to 0 mean the
  reported risk matches observed outcomes more closely.
- **ECE**: average absolute calibration error in bins; lower values indicate that
  predicted probabilities align with real-world frequencies.

## Metric Selection Guide

### By Use Case

| Use Case | Primary Metric | Secondary |
|----------|----------------|-----------|
| Disease screening | Recall | AUC-PRC |
| Spam detection | Precision | F1 |
| General classification | F1, MCC | Balanced Acc |
| Ranking | AUC-ROC | AUC-PRC |
| Risk assessment | Brier, ECE | AUC-ROC |

### By Class Balance

| Balance | Recommended | Avoid |
|---------|-------------|-------|
| Balanced | Accuracy, F1, AUC | - |
| Moderate imbalance | Balanced Acc, F1, MCC | Accuracy |
| Severe imbalance | AUC-PRC, MCC | Accuracy, AUC-ROC alone |

## Confidence Intervals

Always report uncertainty:

```python
# Bootstrap 95% CI
from scipy import stats

def bootstrap_metric(y_true, y_pred, metric_fn, n=1000):
    scores = []
    for _ in range(n):
        idx = np.random.randint(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])

ci = bootstrap_metric(y_true, y_pred, accuracy_score)
print(f"Accuracy: 0.85 (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
```

## Reporting Guidelines

### Minimum to Report

1. **Dataset**: Sample size, class distribution
2. **Primary metric**: AUC or F1 with CI
3. **Secondary metrics**: Accuracy, precision, recall
4. **Confusion matrix**: For interpretability

### For Imbalanced Data

1. **AUC-PRC** alongside AUC-ROC
2. **Per-class metrics** (precision, recall per class)
3. **Balanced accuracy** or MCC
4. **Class distribution** before/after any resampling

### Example

> The model achieved AUC-ROC of 0.923 (95% CI: 0.901-0.945) and AUC-PRC of 0.782 (0.741-0.823) on the held-out test set. Given the class imbalance (90:10), we report balanced accuracy of 0.831 and MCC of 0.712. Per-class recall was 0.95 (majority) and 0.71 (minority).
