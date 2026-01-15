# train-meta

Train a multiclass classifier using binary task ensemble with meta-classifier.

## Usage

```bash
classiflow train-meta [OPTIONS]
```

## Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV file with features and labels |
| `--label-col TEXT` | Name of the label column |
| `--classes TEXT...` | List of class labels |

## Optional Options

### Task Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tasks-json PATH` | None | JSON file with custom task definitions |
| `--tasks-only` | False | Use only tasks from JSON (skip auto OvR/pairwise) |

### Cross-Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outer-folds INT` | 3 | Number of outer CV folds |
| `--inner-splits INT` | 5 | Number of inner CV splits |
| `--inner-repeats INT` | 2 | Number of inner CV repeats |
| `--random-state INT` | 42 | Random seed |

### SMOTE Options

| Option | Default | Description |
|--------|---------|-------------|
| `--smote [off\|on\|both]` | `off` | SMOTE mode |
| `--smote-k INT` | 5 | SMOTE k_neighbors |

### Meta-Classifier Options

| Option | Default | Description |
|--------|---------|-------------|
| `--meta-c TEXT` | `0.01,0.1,1,10` | Comma-separated C values for meta LR |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | `derived` | Output directory |
| `--verbose` | False | Verbose output |

## Examples

### Basic Usage

```bash
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outdir derived/meta
```

### With Custom Tasks

```bash
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC TypeD \
  --tasks-json custom_tasks.json \
  --outdir derived/meta_custom
```

### Full Configuration

```bash
classiflow train-meta \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outer-folds 5 \
  --inner-splits 5 \
  --smote both \
  --meta-c 0.001,0.01,0.1,1,10 \
  --outdir derived/meta_full \
  --verbose
```

## Custom Tasks JSON Format

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

## Auto-Generated Tasks

Without `--tasks-only`, classiflow generates:

- **OvR tasks**: One per class (e.g., `TypeA_vs_Rest`)
- **Pairwise tasks**: One per class pair (e.g., `TypeA_vs_TypeB`)

For 3 classes: 3 OvR + 3 pairwise = 6 tasks

## Output Structure

```
outdir/
├── run.json
├── config.json
├── fold_1/
│   ├── binary_none/        # Binary task results
│   │   ├── binary_pipes.joblib
│   │   └── ...
│   ├── meta_none/          # Meta-classifier
│   │   ├── meta_model.joblib
│   │   └── meta_results.csv
│   └── metrics_outer_eval.csv
└── summary_metrics.csv
```

## See Also

- [train-binary](train-binary.md) - Single binary task
- [train-hierarchical](train-hierarchical.md) - Two-level classification
- [Multiclass Tutorial](../tutorials/multiclass-classification.md)
