# What is a Run

A **run** is a complete execution of a classiflow training pipeline, including all artifacts needed for reproducibility and inference.

## Run Components

```
derived/my_run/
├── run.json                    # Run manifest (lineage)
├── config.json                 # Training configuration
├── fold_1/                     # Outer fold 1
│   ├── binary_none/            # Without SMOTE
│   │   ├── binary_pipes.joblib # Trained pipelines
│   │   ├── best_models.json    # Best model per metric
│   │   └── cv_results.csv      # Inner CV results
│   ├── binary_smote/           # With SMOTE (if enabled)
│   │   └── ...
│   └── metrics_outer_eval.csv  # Outer fold evaluation
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── summary_metrics.csv         # Aggregated metrics
```

## The Run Manifest

The `run.json` file is the heart of reproducibility:

```json
{
  "run_id": "a1b2c3d4-e5f6-...",
  "timestamp": "2024-01-15T10:30:00",
  "package_version": "0.1.0",

  "training_data_path": "data/features.csv",
  "training_data_hash": "sha256:abc123...",
  "training_data_row_count": 569,

  "config": {
    "label_col": "diagnosis",
    "outer_folds": 5,
    "smote_mode": "both"
  },

  "feature_list": ["feature_1", "feature_2", ...],
  "task_definitions": {...},
  "best_models": {"binary_task": "LogisticRegression"},

  "python_version": "3.10.12",
  "hostname": "research-server",
  "git_hash": "abc123def456..."
}
```

### Manifest Fields

| Field | Purpose |
|-------|---------|
| `run_id` | Unique identifier (UUID4) |
| `timestamp` | When training started |
| `training_data_hash` | SHA256 of input data for verification |
| `config` | Complete training configuration |
| `feature_list` | Ordered list of features used |
| `best_models` | Best model selected per task |
| `git_hash` | Repository state (if available) |

## Run Types

### Binary Run

Single binary classification task:
- One task trained per fold
- `binary_none/` and/or `binary_smote/` directories

### Meta Run

Multiclass via binary task ensemble:
- Multiple binary tasks per fold
- Additional `meta_none/` and/or `meta_smote/` directories
- Meta-classifier combines task scores

### Hierarchical Run

Two-level classification:
- L1 classifier for primary categories
- Branch-specific L2 classifiers
- Patient-level stratification

## Using a Run

### Load for Inspection

```python
import json
from pathlib import Path

run_dir = Path("derived/my_run")

# Load manifest
with open(run_dir / "run.json") as f:
    manifest = json.load(f)

print(f"Run ID: {manifest['run_id']}")
print(f"Features: {len(manifest['feature_list'])}")
```

### Load for Inference

```python
from classiflow.inference import run_inference, InferenceConfig

config = InferenceConfig(
    run_dir="derived/my_run",
    data_csv="new_data.csv",
    output_dir="inference_results",
)

results = run_inference(config)
```

### Create Portable Bundle

```bash
classiflow bundle create \
  --run-dir derived/my_run \
  --out my_model.zip
```

## Best Practices

!!! tip "Preserve Run Directories"
    Keep run directories intact. They contain everything needed for reproducibility.

!!! tip "Use Descriptive Output Paths"
    ```bash
    --outdir derived/binary_smote_v2_20240115
    ```

!!! warning "Don't Modify Artifacts"
    Editing files in a run directory breaks reproducibility guarantees.
