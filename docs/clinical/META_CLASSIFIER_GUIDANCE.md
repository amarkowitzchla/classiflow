# Clinical Guidance: Meta-Classifier / Stacking Approach

## Executive Summary

This document provides guidance on when to use meta-classifiers (stacking) in clinical ML assays, based on a critical bug analysis that revealed fundamental issues with the meta-classifier pipeline.

**Key Finding**: The meta-classifier approach is **NOT recommended** as the default for clinical ML tests due to inherent pipeline complexity and risk of silent failures. The multiclass approach should be preferred unless specific conditions justify the added complexity.

---

## Background: The Bug That Revealed Fundamental Issues

### What Happened

A meta-classifier project showed:
- **Technical validation (CV)**: 100% balanced accuracy, 100% F1
- **Independent test**: 10.7% balanced accuracy, 6.2% F1 (near random)

### Root Cause

The `_train_final_meta` function used the **global best** hyperparameter configuration for ALL binary tasks, instead of **per-task best** configurations. This resulted in:

1. Binary classifiers trained with wrong architecture (e.g., `n_layers=3, dropout=0.4` instead of `n_layers=2, dropout=0.2`)
2. Poorly trained models producing near-random predictions (~0.5 probability for all samples)
3. Meta-classifier collapsing to predict only the majority class

### Why This Matters for Clinical Use

This bug demonstrates that:
1. Meta-classifier pipelines have **many more failure modes** than direct multiclass models
2. **Silent failures** can occur where CV looks excellent but deployment fails catastrophically
3. Schema mismatches, hyperparameter drift, and feature order issues are all potential failure points

---

## When Meta-Classifier / Stacking IS Appropriate

### 1. Heterogeneous Modalities
When you have fundamentally different data types that require different preprocessing:
- Example: DNA methylation + RNA expression + clinical features
- Each modality needs its own preprocessing and base learner
- Meta-level combines these heterogeneous predictions

### 2. Established, Locked Pipeline
When the full pipeline has been:
- Validated on multiple independent cohorts
- Frozen with strict schema locking
- Tested for calibration stability
- Deployed with runtime validation checks

### 3. Complementary Base Learners
When base learners capture genuinely different aspects of the problem:
- Example: OvR binary tasks + pairwise comparisons + specialized clinical rules
- Base learners should have low correlation in their errors

### 4. Large Datasets with Many Classes
When you have:
- Sufficient samples per class for stable meta-features (recommend >30 per class)
- Enough total samples for proper nested CV (recommend >500 total)
- Low risk of overfitting at the meta level

---

## When to AVOID Meta-Classifier

### 1. Small Datasets
- **Problem**: Insufficient data for proper nested CV leads to unstable meta-features
- **Threshold**: Avoid with <500 total samples or <20 samples in smallest class
- **Alternative**: Use direct multiclass with appropriate regularization

### 2. High Class Imbalance
- **Problem**: OvR binary classifiers for rare classes have very few positive examples
- **Threshold**: Avoid when any class has <5% prevalence
- **Alternative**: Multiclass with class weights or SMOTE

### 3. Single Modality Data
- **Problem**: No benefit from meta-level aggregation; just adds complexity
- **Recommendation**: Use direct multiclass model
- **Evidence**: In the investigated case, multiclass achieved 91.3% balanced accuracy vs 57.6% for fixed meta-classifier

### 4. Pipeline Drift Risk
- **Problem**: Any change to base learners invalidates the meta-classifier
- **Risk factors**:
  - Frequent retraining
  - Feature set changes
  - Model architecture updates
- **Alternative**: Direct model with continuous monitoring

### 5. Inability to Guarantee Schema Consistency
- **Problem**: Meta-features depend on exact column order and naming
- **Risk**: Feature order mismatch causes silent failures
- **Alternative**: Direct model with explicit feature contracts

---

## Promotion Gates for Meta-Classifiers

If meta-classifier is used, require these additional gates:

### 1. Performance Consistency Gate
```
independent_test_accuracy >= cv_accuracy - 0.15
independent_test_f1_macro >= cv_f1_macro - 0.15
```
Flag if test performance drops >15% from CV.

### 2. Calibration Quality Gate
```
brier_score < 0.3
ece (expected calibration error) < 0.10
```
Poorly calibrated meta-classifiers indicate overfitting.

### 3. Prediction Distribution Gate
```
# All classes should receive some predictions
min_class_prediction_rate > 0.01

# No single class should dominate
max_class_prediction_rate < 0.8
```
Collapsed predictions indicate meta-classifier failure.

### 4. Schema Lock Verification
```
# At inference time, verify:
- meta_features match saved schema exactly
- class order matches saved classes exactly
- probability columns align with class labels
```

### 5. Binary Pipeline Sanity Check
```
# For each binary pipeline:
- prediction_std > 0.05  (not near-random)
- prediction_mean not in [0.45, 0.55]  (not collapsed)
```

---

## Recommended Default: Multiclass

For most clinical ML assays, use **direct multiclass classification**:

### Advantages
1. **Simpler pipeline**: Fewer potential failure points
2. **Better interpretability**: Single model to explain
3. **Easier monitoring**: One model to track
4. **More stable**: No schema mismatch risk

### When Multiclass May Underperform
1. When classes have non-exclusive relationships
2. When different classes need different feature subsets
3. When hierarchical structure exists (consider hierarchical models instead)

---

## Implementation Checklist

### Before Using Meta-Classifier

- [ ] Dataset has >500 samples total
- [ ] Smallest class has >30 samples
- [ ] There is a clear reason why multiclass is insufficient
- [ ] Team has expertise to debug meta-classifier failures
- [ ] Runtime validation will be implemented

### During Development

- [ ] Per-task hyperparameter selection is verified
- [ ] Binary pipelines are validated for discriminative outputs
- [ ] Meta-features have reasonable variance
- [ ] Schema is locked and versioned
- [ ] Independent test is used for final evaluation (never for tuning)

### Before Deployment

- [ ] Performance drop from CV to test is <15%
- [ ] Calibration metrics meet thresholds (Brier <0.3, ECE <0.10)
- [ ] Prediction distribution is reasonable (no class collapse)
- [ ] Runtime validation is implemented and tested
- [ ] Rollback plan exists

---

## Summary

| Criterion | Multiclass | Meta-Classifier |
|-----------|------------|-----------------|
| Default recommendation | **YES** | No |
| Single modality | **YES** | No |
| Small dataset (<500) | **YES** | No |
| Heterogeneous modalities | Consider | **YES** |
| Locked, validated pipeline | Consider | **YES** |
| Clinical deployment | **YES** | Only with extra gates |

**Bottom Line**: Unless you have specific requirements that multiclass cannot meet, use multiclass. The added complexity of meta-classifiers creates more opportunities for silent failures that can be catastrophic in clinical settings.
