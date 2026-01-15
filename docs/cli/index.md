# CLI Reference

Classiflow provides a comprehensive command-line interface for all major workflows.

## Installation

The CLI is automatically available after installing classiflow:

```bash
pip install classiflow
classiflow --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| [`train-binary`](train-binary.md) | Binary classification with nested CV |
| [`train-meta`](train-meta.md) | Multiclass via binary task ensemble |
| [`train-hierarchical`](train-hierarchical.md) | Hierarchical classification |
| [`infer`](infer.md) | Run inference on new data |
| [`stats`](stats.md) | Statistical analysis and visualization |
| [`bundle`](bundle.md) | Create and manage model bundles |

## Global Options

```bash
classiflow --version   # Show version
classiflow --help      # Show help
```

## Common Patterns

### Training Workflow

```bash
# Train binary classifier
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Malignant \
  --outdir derived/binary

# Train multiclass
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outdir derived/meta
```

### Inference Workflow

```bash
# Run inference
classiflow infer \
  --run-dir derived/binary \
  --data-csv data/new_samples.csv \
  --outdir derived/inference
```

### Statistical Analysis

```bash
# Run stats
classiflow stats run \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --outdir derived/stats

# Generate visualizations
classiflow stats viz \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --stats-dir derived/stats
```

### Model Bundling

```bash
# Create bundle
classiflow bundle create \
  --run-dir derived/binary \
  --out model.zip

# Inspect bundle
classiflow bundle inspect model.zip
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLASSIFLOW_LOG_LEVEL` | Logging verbosity | `INFO` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (see stderr for details) |
| 2 | Invalid arguments |

## Getting Help

Each command has its own help:

```bash
classiflow train-binary --help
classiflow infer --help
classiflow stats --help
```
