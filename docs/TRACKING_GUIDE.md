# Experiment Tracking Guide

classiflow supports optional integration with experiment tracking systems to help you manage and visualize your machine learning experiments. This guide covers setup and usage of MLflow and Weights & Biases (W&B) tracking.

## Installation

Tracking backends are optional dependencies. Install the one you need:

```bash
# For MLflow
pip install classiflow[mlflow]

# For Weights & Biases
pip install classiflow[wandb]

# For both
pip install classiflow[tracking]
```

## Quick Start

### CLI Usage

Add tracking to any training command with `--tracker`:

```bash
# Binary training with MLflow
classiflow train-binary \
    --data data.parquet \
    --label-col diagnosis \
    --outdir results/binary \
    --tracker mlflow \
    --experiment-name my-experiment

# Meta-classifier training with W&B
classiflow train-meta \
    --data data.parquet \
    --label-col subtype \
    --outdir results/meta \
    --tracker wandb \
    --experiment-name my-project \
    --run-name meta-v1
```

### Available Options

| Option | Description |
|--------|-------------|
| `--tracker` | Tracking backend: `mlflow` or `wandb` |
| `--experiment-name` | Experiment/project name (default varies by command) |
| `--run-name` | Run name (auto-generated if not specified) |

## What Gets Tracked

### Parameters

All training configuration is logged as parameters:
- Data paths, label columns
- Cross-validation settings (outer_folds, inner_splits, etc.)
- SMOTE configuration
- Model backend and device settings
- Calibration settings (for meta-classifier)

### Metrics

Summary metrics from training are logged:
- Accuracy, balanced accuracy
- F1 scores (macro, weighted)
- ROC AUC
- MCC, sensitivity, specificity
- Calibration metrics (Brier score, ECE) for meta-classifier

### Artifacts

Training artifacts are automatically logged:
- `run.json` - Training manifest with full lineage
- `metrics_*.csv` - All metrics tables
- `*.png` - ROC curves, PR curves, confusion matrices

### Tags

Runs are tagged with metadata for easy filtering:
- `task_type`: binary, meta, multiclass, or hierarchical
- `backend`: sklearn or torch
- `smote_mode`: off, on, or both
- `run_id`: Links to classiflow manifest

## MLflow Setup

### Local Tracking (Default)

By default, MLflow stores runs in a local `./mlruns` directory:

```bash
# Run training
classiflow train-binary --data data.csv --label-col label --tracker mlflow

# View results
mlflow ui
# Open http://localhost:5000
```

### Remote Tracking Server

Set the `MLFLOW_TRACKING_URI` environment variable:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
classiflow train-binary --data data.csv --label-col label --tracker mlflow
```

### MLflow Concepts

- **Experiment**: Groups related runs (e.g., "my-project")
- **Run**: Single training execution
- **Artifacts**: Files associated with a run (metrics, models, plots)

## Weights & Biases Setup

### Authentication

Log in to W&B before first use:

```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Running with W&B

```bash
classiflow train-binary \
    --data data.csv \
    --label-col label \
    --tracker wandb \
    --experiment-name my-project
```

View runs at: https://wandb.ai/your-username/my-project

### W&B Concepts

- **Project**: Groups related runs (equivalent to MLflow experiment)
- **Run**: Single training execution
- **Artifacts**: Versioned files (metrics, models)
- **Tags**: Labels for filtering runs

## Project Workflow Integration

Tracking integrates with classiflow project workflows:

### Configuration in project.yaml

Add tracking settings to your project configuration:

```yaml
project:
  id: my-classifier
  name: My Classifier

# ... other config ...

# Tracking (optional)
tracker: mlflow  # or wandb
experiment_name: my-classifier  # optional, defaults to {project_id}/{phase}
```

### Command-Line Override

Override project settings via CLI:

```bash
classiflow project run-technical ./my-project \
    --tracker mlflow \
    --experiment-name custom-experiment
```

### Experiment Organization

When using project workflows, runs are automatically organized:
- **MLflow**: Uses nested experiments (`{project_id}/technical`, `{project_id}/final`, etc.)
- **W&B**: Uses groups to organize runs by phase

## Programmatic Usage

Use tracking directly in Python code:

```python
from classiflow.tracking import get_tracker

# Get a tracker instance
tracker = get_tracker(
    backend="mlflow",  # or "wandb" or None for no-op
    experiment_name="my-experiment",
)

# Use as context manager
with tracker.start_run(run_name="run-1", tags={"version": "1.0"}):
    tracker.log_params({"lr": 0.01, "epochs": 100})

    # ... training code ...

    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    tracker.log_artifact("model.pkl")
```

### NoOpTracker

When tracking is disabled (default), a `NoOpTracker` is used that silently ignores all tracking calls. This ensures your code works identically with or without tracking:

```python
tracker = get_tracker(backend=None)  # Returns NoOpTracker
tracker.log_params({"a": 1})  # Does nothing, no errors
```

## Best Practices

1. **Use consistent experiment names** across related runs for easy comparison
2. **Add meaningful run names** to identify specific configurations
3. **Use tags** for metadata that you'll want to filter on later
4. **Don't duplicate lineage data** - classiflow's `run.json` already captures data hashes and git commits
5. **Review artifacts** - the logged CSV and PNG files are useful for debugging

## Troubleshooting

### MLflow not found

```
ImportError: MLflow is not installed. Install with: pip install classiflow[mlflow]
```

Install the mlflow optional dependency.

### W&B not authenticated

```
wandb: ERROR Run data directory already exists
```

Run `wandb login` to authenticate.

### Tracking not working

Check that:
1. The tracker is correctly specified (`--tracker mlflow` or `--tracker wandb`)
2. The tracking backend is installed
3. For MLflow: check `./mlruns` directory exists
4. For W&B: ensure you're logged in and have network access
