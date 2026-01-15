# Pipeline Boundaries

Understanding what happens inside each cross-validation fold is critical for valid results.

## The Data Leakage Problem

**Data leakage** occurs when information from the test set influences training, leading to overoptimistic performance estimates.

### Common Leakage Sources

1. **Scaling on full data** before splitting
2. **SMOTE on full data** before splitting
3. **Feature selection on full data** before splitting
4. **Hyperparameter tuning** on the test set

## Classiflow's Solution

All data-dependent transformations occur **inside the training portion** of each fold:

```
Outer Fold i:
┌─────────────────────────────────────────────────────────────┐
│ TRAINING DATA (folds j ≠ i)                                │
│                                                            │
│   1. Feature standardization (fit + transform)             │
│   2. SMOTE (if enabled)                                    │
│   3. Inner CV for hyperparameter tuning                    │
│   4. Final model training                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Trained Pipeline
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ TEST DATA (fold i)                                          │
│                                                            │
│   1. Feature standardization (transform only)              │
│   2. Prediction                                            │
│   3. Metric computation                                    │
└─────────────────────────────────────────────────────────────┘
```

## What Happens Inside Each Fold

### 1. Feature Standardization

```python
from sklearn.preprocessing import StandardScaler

# Inside training fold:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train

# On test fold:
X_test_scaled = scaler.transform(X_test)  # Transform only
```

!!! warning "Never fit on test data"
    The scaler parameters (mean, std) come only from training data.

### 2. SMOTE Application

```python
from imblearn.over_sampling import SMOTE

# Inside training fold only:
smote = SMOTE(k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Test fold is never resampled
```

!!! warning "SMOTE creates synthetic samples"
    Applying SMOTE before splitting means synthetic test samples may be based on training samples—a severe form of leakage.

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Inner CV uses only training data:
inner_cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
grid_search.fit(X_train, y_train)  # Only training data
```

### 4. Model Training

```python
# Train final model on full training fold:
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
```

## The Scikit-learn Pipeline

Classiflow uses sklearn pipelines to ensure correct ordering:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE()),  # imblearn-compatible
    ("classifier", LogisticRegression())
])

# The pipeline ensures:
# 1. Scaler fits on training data only
# 2. SMOTE applies to training data only
# 3. Classifier trains on processed data
```

## Why Nested CV?

### Single-Level CV Problem

```python
# WRONG: Single-level CV
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X, y)
print(grid_search.best_score_)  # Biased!
```

The `best_score_` is optimistic because:
- We searched for parameters that maximize this score
- We report the same score as our final estimate

### Nested CV Solution

```python
# CORRECT: Nested CV
outer_cv = StratifiedKFold(n_splits=5)  # For evaluation
inner_cv = StratifiedKFold(n_splits=5)  # For tuning

scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Tune on training data
    grid_search = GridSearchCV(model, params, cv=inner_cv)
    grid_search.fit(X_train, y_train)

    # Evaluate on held-out test
    score = grid_search.score(X_test, y_test)
    scores.append(score)

print(f"Unbiased estimate: {np.mean(scores):.3f}")
```

## Hierarchical Classification Boundaries

For hierarchical models with patient-level data:

```
Patient-Level Split (no patient in both train and test)
├── L1 Training (tumor type classifier)
│   └── Uses patient samples from training patients only
├── L2 Training (branch-specific subtype classifiers)
│   └── Uses samples from training patients in each branch
└── Evaluation
    └── Uses samples from held-out test patients
```

!!! warning "Patient Leakage"
    If the same patient has multiple samples, all must be in the same fold. Otherwise, the model can "recognize" the patient rather than learning general patterns.

## Summary

| Transformation | Applied To |
|----------------|------------|
| StandardScaler.fit | Training only |
| SMOTE | Training only |
| Feature selection | Training only |
| Hyperparameter tuning | Training only |
| StandardScaler.transform | Both |
| Prediction | Test only |
| Metric computation | Test only |

## References

- Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079-2107.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7(1), 91.
