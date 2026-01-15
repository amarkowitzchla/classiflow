# Configure a Run

This guide covers all configuration options for training runs.

## Configuration Classes

Classiflow provides three main configuration classes:

| Class | Use Case |
|-------|----------|
| `TrainConfig` | Binary classification |
| `MetaConfig` | Multiclass via binary ensembles |
| `HierarchicalConfig` | Two-level hierarchical classification |

## TrainConfig (Binary)

```python
from classiflow import TrainConfig

config = TrainConfig(
    # Required
    data_csv="data/features.csv",  # Path to CSV
    label_col="diagnosis",         # Label column name

    # Binary task
    pos_label="Malignant",         # Positive class (default: minority)
    feature_cols=None,             # Explicit features (default: all numeric)

    # Output
    outdir="derived/run",          # Output directory

    # Cross-validation
    outer_folds=5,                 # Outer CV folds (default: 3)
    inner_splits=5,                # Inner CV folds (default: 5)
    inner_repeats=2,               # Inner CV repeats (default: 2)
    random_state=42,               # Random seed (default: 42)

    # SMOTE
    smote_mode="off",              # "off", "on", or "both"
    smote_k_neighbors=5,           # SMOTE k parameter

    # Models
    max_iter=10000,                # Solver max iterations
)
```

### Feature Selection

```python
# Auto-detect numeric columns
config = TrainConfig(
    data_csv="data.csv",
    label_col="target",
    feature_cols=None,  # All numeric columns except label
)

# Explicit feature list
config = TrainConfig(
    data_csv="data.csv",
    label_col="target",
    feature_cols=["feature_1", "feature_2", "feature_3"],
)
```

## MetaConfig (Multiclass)

Extends `TrainConfig` with multiclass options:

```python
from classiflow import MetaConfig

config = MetaConfig(
    # All TrainConfig options, plus:

    # Class specification
    classes=["TypeA", "TypeB", "TypeC"],  # Required for multiclass

    # Task definition
    tasks_json="custom_tasks.json",  # Optional custom tasks
    tasks_only=False,                # If True, skip auto OvR/pairwise

    # Meta-classifier
    meta_C_grid=[0.01, 0.1, 1, 10],  # Regularization search
)
```

### Custom Tasks JSON

```json
[
  {
    "name": "TypeA_vs_TypeB",
    "pos": ["TypeA"],
    "neg": ["TypeB"]
  },
  {
    "name": "Rare_vs_Common",
    "pos": ["TypeC"],
    "neg": "rest"
  }
]
```

## HierarchicalConfig

For two-level classification:

```python
from classiflow.config import HierarchicalConfig

config = HierarchicalConfig(
    # Data
    data_csv="data/features.csv",
    patient_col="patient_id",      # For patient-level stratification
    label_l1="tumor_type",         # Level-1 label
    label_l2="subtype",            # Level-2 label (optional)
    l2_classes=None,               # Subset of L2 classes
    min_l2_classes_per_branch=2,   # Min classes for branch model
    feature_cols=None,

    # Output
    outdir="derived/hierarchical",
    output_format="xlsx",          # "xlsx" or "csv"

    # Cross-validation
    outer_folds=5,
    inner_splits=3,
    random_state=42,

    # PyTorch MLP
    device="auto",                 # "auto", "cpu", "cuda", "mps"
    mlp_epochs=100,
    mlp_batch_size=256,
    mlp_hidden=128,
    mlp_dropout=0.3,
    early_stopping_patience=10,

    # SMOTE
    use_smote=False,
    smote_k_neighbors=5,

    # Logging
    verbose=1,                     # 0, 1, or 2
)
```

## CLI Configuration

### Binary Training

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Malignant \
  --outer-folds 5 \
  --inner-splits 5 \
  --inner-repeats 2 \
  --smote both \
  --random-state 42 \
  --outdir derived/binary \
  --verbose
```

### Meta-Classifier

```bash
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outer-folds 5 \
  --smote off \
  --outdir derived/meta
```

### Hierarchical

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --device auto \
  --use-smote \
  --outer-folds 5 \
  --mlp-epochs 100 \
  --outdir derived/hierarchical \
  --verbose 2
```

## Saving and Loading Configs

```python
# Save config to JSON
config.save("config.json")

# Load and recreate (manual)
import json
with open("config.json") as f:
    config_dict = json.load(f)

config = TrainConfig(**config_dict)
```

## Configuration Validation

Configs validate on creation:

```python
try:
    config = TrainConfig(
        data_csv="nonexistent.csv",  # Will raise error
        label_col="diagnosis",
    )
except FileNotFoundError as e:
    print(f"Validation error: {e}")
```

## Best Practices

!!! tip "Start with Defaults"
    The defaults are sensible for most cases. Only override what you need.

!!! tip "Use `smote_mode='both'`"
    For imbalanced data, train both variants to compare.

!!! tip "Set `random_state`"
    Always set for reproducibility, even if you don't plan to reproduce.

!!! warning "Watch Memory with Large Datasets"
    Reduce `inner_repeats` or `outer_folds` if running out of memory.
