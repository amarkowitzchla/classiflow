# train-binary

Train a binary classification model with nested cross-validation.

## Usage

```bash
classiflow train-binary [OPTIONS]
```

## Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV file with features and labels |
| `--label-col TEXT` | Name of the label column |

## Optional Options

### Data Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pos-label TEXT` | Auto | Positive class label (default: minority class) |
| `--feature-cols TEXT` | All numeric | Comma-separated feature column names |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | `derived` | Output directory for artifacts |

### Cross-Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outer-folds INT` | 3 | Number of outer CV folds |
| `--inner-splits INT` | 5 | Number of inner CV splits |
| `--inner-repeats INT` | 2 | Number of inner CV repeats |
| `--random-state INT` | 42 | Random seed for reproducibility |

### SMOTE Options

| Option | Default | Description |
|--------|---------|-------------|
| `--smote [off\|on\|both]` | `off` | SMOTE mode |
| `--smote-k INT` | 5 | SMOTE k_neighbors parameter |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-iter INT` | 10000 | Max iterations for linear models |

### Logging Options

| Option | Description |
|--------|-------------|
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress non-error output |

## Examples

### Basic Usage

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Malignant \
  --outdir derived/binary
```

### With SMOTE

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Rare_Class \
  --smote both \
  --smote-k 5 \
  --outdir derived/binary_smote
```

### Full Configuration

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label Malignant \
  --outer-folds 5 \
  --inner-splits 5 \
  --inner-repeats 2 \
  --random-state 42 \
  --smote both \
  --max-iter 10000 \
  --outdir derived/binary_full \
  --verbose
```

## Output Structure

```
outdir/
├── run.json              # Run manifest
├── config.json           # Configuration
├── fold_1/
│   ├── binary_none/      # Without SMOTE
│   │   ├── binary_pipes.joblib
│   │   ├── best_models.json
│   │   └── cv_results.csv
│   ├── binary_smote/     # With SMOTE (if enabled)
│   │   └── ...
│   └── metrics_outer_eval.csv
├── fold_2/
│   └── ...
└── summary_metrics.csv
```

## See Also

- [train-meta](train-meta.md) - Multiclass classification
- [infer](infer.md) - Run inference on trained models
- [Binary Classification Tutorial](../tutorials/binary-classification.md)
