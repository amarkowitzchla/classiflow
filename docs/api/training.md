# Training API

Main entry points for model training.

## Module: `classiflow`

The top-level module exports the main training functions.

```python
from classiflow import train_binary_task, train_meta_classifier, TaskBuilder
from classiflow import TrainConfig, MetaConfig
```

---

## train_binary_task

::: classiflow.training.binary.train_binary_task
    options:
      show_root_heading: true
      show_source: true

### Example

```python
from classiflow import train_binary_task, TrainConfig

config = TrainConfig(
    data_csv="data/features.csv",
    label_col="diagnosis",
    pos_label="Malignant",
    outer_folds=5,
    inner_splits=5,
    inner_repeats=2,
    smote_mode="both",
    outdir="derived/binary",
)

results = train_binary_task(config)

# Access results
print(f"Mean AUC: {results['summary']['roc_auc']['mean']:.3f}")
```

### Returns

```python
{
    "run_id": "uuid-string",
    "task_results": {
        "binary_task": {
            "fold_1": {...},
            "fold_2": {...},
            ...
        }
    },
    "summary": {
        "roc_auc": {"mean": 0.92, "std": 0.03, ...},
        "accuracy": {...},
        ...
    }
}
```

---

## train_meta_classifier

::: classiflow.training.meta.train_meta_classifier
    options:
      show_root_heading: true
      show_source: true

### Example

```python
from classiflow import train_meta_classifier, MetaConfig

config = MetaConfig(
    data_csv="data/features.csv",
    label_col="subtype",
    classes=["TypeA", "TypeB", "TypeC"],
    outer_folds=5,
    smote_mode="off",
    outdir="derived/meta",
)

results = train_meta_classifier(config)
```

---

## Hierarchical Training

For hierarchical classification, use the config-based approach:

```python
from classiflow.config import HierarchicalConfig
from classiflow.training.hierarchical_cv import train_hierarchical

config = HierarchicalConfig(
    data_csv="data/features.csv",
    patient_col="patient_id",
    label_l1="tumor_type",
    label_l2="subtype",
    device="auto",
    use_smote=True,
    outdir="derived/hierarchical",
)

results = train_hierarchical(config)
```

---

## See Also

- [TrainConfig](config.md#trainconfig) - Binary training configuration
- [MetaConfig](config.md#metaconfg) - Multiclass training configuration
- [HierarchicalConfig](config.md#hierarchicalconfig) - Hierarchical training configuration
