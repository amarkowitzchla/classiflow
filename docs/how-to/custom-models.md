# Add Custom Models

This guide shows how to integrate custom scikit-learn compatible models.

## Model Requirements

Custom models must implement the scikit-learn estimator interface:

- `fit(X, y)` - Train the model
- `predict(X)` - Return class predictions
- `predict_proba(X)` - Return probability estimates (for AUC calculation)

## Current Model Set

Classiflow includes these models by default:

```python
from classiflow.models import get_estimators, get_param_grids

# Get default estimators
estimators = get_estimators(max_iter=10000)
print("Default models:", list(estimators.keys()))
# ['LogisticRegression', 'SVC', 'RandomForest', 'GradientBoosting']

# Get hyperparameter grids
param_grids = get_param_grids()
for name, grid in param_grids.items():
    print(f"{name}: {grid}")
```

## Adding Models via Python

### Modify the Estimator Registry

Create a custom training script that extends the default models:

```python
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from classiflow.models import get_estimators, get_param_grids
from classiflow.training.nested_cv import NestedCVOrchestrator

def get_custom_estimators(max_iter=10000):
    """Get estimators with custom additions."""
    estimators = get_estimators(max_iter=max_iter)

    # Add custom models
    estimators["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    estimators["AdaBoost"] = AdaBoostClassifier(
        n_estimators=50,
        random_state=42,
    )
    estimators["KNN"] = KNeighborsClassifier(
        n_neighbors=5,
    )

    return estimators

def get_custom_param_grids():
    """Get param grids with custom additions."""
    grids = get_param_grids()

    grids["ExtraTrees"] = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    grids["AdaBoost"] = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 1.0],
    }
    grids["KNN"] = {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    }

    return grids

# Use in training (requires modifying NestedCVOrchestrator)
```

### Using a Custom Orchestrator

```python
from classiflow.training.nested_cv import NestedCVOrchestrator

class CustomNestedCV(NestedCVOrchestrator):
    """Custom orchestrator with additional models."""

    def _get_estimators(self):
        """Override to include custom models."""
        estimators = super()._get_estimators()

        from sklearn.neural_network import MLPClassifier

        estimators["MLP"] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
        )
        return estimators

    def _get_param_grids(self):
        """Override to include custom param grids."""
        grids = super()._get_param_grids()

        grids["MLP"] = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
        }
        return grids
```

## Adding Deep Learning Models

For PyTorch models, use the existing `TorchMLP`:

```python
from classiflow.models import TorchMLP, TorchMLPWrapper

# Create a custom architecture
class CustomMLP(TorchMLPWrapper):
    """Wrapper for sklearn compatibility."""

    def __init__(self, hidden_dim=128, dropout=0.3, **kwargs):
        super().__init__(
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
```

## Model Selection Criteria

Classiflow selects models based on inner CV performance:

```python
# Default scoring metrics for model selection
scoring = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}

# Primary metric for selection (default: roc_auc)
refit_metric = "roc_auc"
```

## Example: XGBoost Integration

```python
# Requires: pip install xgboost

from xgboost import XGBClassifier

def get_estimators_with_xgb(max_iter=10000):
    """Get estimators including XGBoost."""
    from classiflow.models import get_estimators

    estimators = get_estimators(max_iter=max_iter)
    estimators["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    return estimators

param_grids_xgb = {
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }
}
```

## Best Practices

!!! tip "Ensure Reproducibility"
    Always set `random_state` in custom models.

!!! tip "Support `predict_proba`"
    Models must return probabilities for AUC calculation.

!!! warning "Scaling Requirements"
    Some models (SVM, KNN) require scaled features. Classiflow applies StandardScaler by default in the pipeline.

!!! warning "Memory Considerations"
    Large hyperparameter grids increase memory usage. Start small.
