# Configuration API

Configuration classes for training and inference.

## Module: `classiflow.config`

```python
from classiflow import TrainConfig, MetaConfig
from classiflow.config import HierarchicalConfig, RunManifest
```

---

## TrainConfig

::: classiflow.config.TrainConfig
    options:
      show_root_heading: true
      members:
        - __init__
        - to_dict
        - save

### Example

```python
from classiflow import TrainConfig

config = TrainConfig(
    # Required
    data_csv="data/features.csv",
    label_col="diagnosis",

    # Binary task
    pos_label="Malignant",       # Positive class (default: minority)
    feature_cols=None,           # None = auto-detect numeric

    # Output
    outdir="derived/run",

    # Cross-validation
    outer_folds=5,
    inner_splits=5,
    inner_repeats=2,
    random_state=42,

    # SMOTE
    smote_mode="both",           # "off", "on", or "both"
    smote_k_neighbors=5,

    # Models
    max_iter=10000,
)

# Save configuration
config.save("config.json")

# Convert to dict
config_dict = config.to_dict()
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_csv` | `Path` | Required | Path to data CSV |
| `label_col` | `str` | Required | Label column name |
| `pos_label` | `str \| None` | `None` | Positive class (auto: minority) |
| `feature_cols` | `list[str] \| None` | `None` | Feature columns (auto: numeric) |
| `outdir` | `Path` | `"derived"` | Output directory |
| `outer_folds` | `int` | `3` | Outer CV folds |
| `inner_splits` | `int` | `5` | Inner CV splits |
| `inner_repeats` | `int` | `2` | Inner CV repeats |
| `random_state` | `int` | `42` | Random seed |
| `smote_mode` | `str` | `"off"` | SMOTE mode |
| `smote_k_neighbors` | `int` | `5` | SMOTE k parameter |
| `max_iter` | `int` | `10000` | Solver iterations |

---

## MetaConfig

::: classiflow.config.MetaConfig
    options:
      show_root_heading: true

Extends `TrainConfig` with multiclass options.

### Example

```python
from classiflow import MetaConfig

config = MetaConfig(
    # All TrainConfig options plus:
    classes=["TypeA", "TypeB", "TypeC"],
    tasks_json="custom_tasks.json",  # Optional
    tasks_only=False,                # Skip auto tasks
    meta_C_grid=[0.01, 0.1, 1, 10],  # Meta LR regularization
)
```

### Additional Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `classes` | `list[str] \| None` | `None` | Class labels |
| `tasks_json` | `Path \| None` | `None` | Custom tasks JSON |
| `tasks_only` | `bool` | `False` | Use only JSON tasks |
| `meta_C_grid` | `list[float]` | `[0.01, 0.1, 1, 10]` | Meta-classifier C values |

---

## MulticlassConfig

::: classiflow.config.MulticlassConfig
    options:
      show_root_heading: true

Extends `TrainConfig` with multiclass options.

### Example

```python
from classiflow import MulticlassConfig

config = MulticlassConfig(
    classes=["TypeA", "TypeB", "TypeC"],
    patient_col="patient_id",
    group_stratify=True,
    logreg_max_iter=5000,
    logreg_tol=1e-3,
)
```

`logreg_multi_class="auto"` selects multinomial behavior for multiclass problems when using `solver="saga"`.

### Additional Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `classes` | `list[str] \| None` | `None` | Class labels |
| `group_stratify` | `bool` | `True` | Stratified group splits when `patient_col` is set |
| `logreg_solver` | `str` | `"saga"` | LogisticRegression solver |
| `logreg_multi_class` | `str` | `"auto"` | LogisticRegression multi_class |
| `logreg_penalty` | `str` | `"l2"` | LogisticRegression penalty |
| `logreg_max_iter` | `int` | `5000` | LogisticRegression max_iter |
| `logreg_tol` | `float` | `1e-3` | LogisticRegression tolerance |
| `logreg_C` | `float` | `1.0` | LogisticRegression C |
| `logreg_class_weight` | `str \| None` | `"balanced"` | LogisticRegression class_weight |
| `logreg_n_jobs` | `int` | `-1` | LogisticRegression n_jobs |

---

## HierarchicalConfig

::: classiflow.config.HierarchicalConfig
    options:
      show_root_heading: true

### Example

```python
from classiflow.config import HierarchicalConfig

config = HierarchicalConfig(
    # Data
    data_csv="data/features.csv",
    patient_col="patient_id",
    label_l1="tumor_type",
    label_l2="subtype",
    l2_classes=None,              # All L2 classes
    min_l2_classes_per_branch=2,

    # Output
    outdir="derived/hierarchical",
    output_format="xlsx",

    # Cross-validation
    outer_folds=5,
    inner_splits=3,
    random_state=42,

    # PyTorch MLP
    device="auto",
    mlp_epochs=100,
    mlp_batch_size=256,
    mlp_hidden=128,
    mlp_dropout=0.3,
    early_stopping_patience=10,

    # SMOTE
    use_smote=True,
    smote_k_neighbors=5,

    # Logging
    verbose=1,
)
```

### Properties

```python
config.hierarchical  # bool: True if label_l2 is set
```

---

## RunManifest

::: classiflow.config.RunManifest
    options:
      show_root_heading: true

### Example

```python
from classiflow.config import RunManifest, TrainConfig

config = TrainConfig(...)
manifest = RunManifest.from_config(config)
manifest.save("run_manifest.json")
```

---

## See Also

- [InferenceConfig](inference.md#inferenceconfig) - Inference configuration
- [StatsConfig](stats.md#statsconfig) - Statistics configuration
