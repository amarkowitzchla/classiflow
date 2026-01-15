# API Reference

Complete documentation of classiflow's Python API.

## Overview

Classiflow is organized into these main modules:

| Module | Purpose |
|--------|---------|
| [`classiflow`](training.md) | Main package exports: `train_binary_task`, `train_meta_classifier` |
| [`classiflow.config`](config.md) | Configuration classes |
| [`classiflow.inference`](inference.md) | Inference pipeline |
| [`classiflow.tasks`](tasks.md) | Task building utilities |
| [`classiflow.stats`](stats.md) | Statistical analysis |
| [`classiflow.bundles`](bundles.md) | Model bundling |
| [`classiflow.io`](io.md) | Data loading and validation |
| [`classiflow.lineage`](lineage.md) | Reproducibility tracking |

## Quick Reference

### Training

```python
from classiflow import train_binary_task, train_meta_classifier, TrainConfig, MetaConfig

# Binary classification
config = TrainConfig(
    data_csv="data.csv",
    label_col="diagnosis",
    pos_label="Malignant",
)
results = train_binary_task(config)

# Multiclass
config = MetaConfig(
    data_csv="data.csv",
    label_col="subtype",
    classes=["A", "B", "C"],
)
results = train_meta_classifier(config)
```

### Inference

```python
from classiflow.inference import run_inference, InferenceConfig

config = InferenceConfig(
    run_dir="derived/run",
    data_csv="new_data.csv",
    output_dir="results",
)
results = run_inference(config)
```

### Statistics

```python
from classiflow.stats import run_stats, run_visualizations, StatsConfig

results = run_stats(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="stats_results",
)
```

### Task Building

```python
from classiflow.tasks import TaskBuilder

builder = TaskBuilder(classes=["A", "B", "C"])
builder.build_all_auto_tasks()
builder.add_composite_task("A_vs_BC", pos_classes=["A"], neg_classes="rest")
tasks = builder.get_tasks()
```

## Type Hints

Classiflow uses type hints throughout. The package includes `py.typed` for PEP 561 support:

```python
from classiflow import TrainConfig
from typing import reveal_type

config = TrainConfig(...)
reveal_type(config)  # TrainConfig
```

## Versioning

```python
import classiflow
print(classiflow.__version__)  # "0.1.0"
```
