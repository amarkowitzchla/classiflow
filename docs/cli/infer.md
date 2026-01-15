# infer

Run inference on new data using trained models.

## Usage

```bash
classiflow infer [OPTIONS]
```

## Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV with samples to classify |
| `--run-dir PATH` | Path to trained run directory |

## Optional Options

### Data Options

| Option | Default | Description |
|--------|---------|-------------|
| `--id-col TEXT` | None | Sample ID column |
| `--label-col TEXT` | None | Ground truth column (for metrics) |

### Feature Alignment

| Option | Default | Description |
|--------|---------|-------------|
| `--strict` | True | Fail on missing features |
| `--lenient` | False | Fill missing features |
| `--fill-strategy [zero\|median]` | `zero` | Fill strategy for lenient mode |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | Required | Output directory |
| `--no-plots` | False | Skip plot generation |
| `--no-excel` | False | Skip Excel report |
| `--max-roc-curves INT` | 10 | Max per-class ROC curves |

### Compute Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device [auto\|cpu\|cuda\|mps]` | `auto` | Compute device |
| `--batch-size INT` | 512 | Batch size |

## Examples

### Basic Inference

```bash
classiflow infer \
  --data-csv data/new_samples.csv \
  --run-dir derived/binary \
  --outdir derived/inference
```

### With Evaluation

```bash
classiflow infer \
  --data-csv data/test_samples.csv \
  --run-dir derived/binary \
  --label-col diagnosis \
  --outdir derived/inference_eval
```

### Lenient Feature Matching

```bash
classiflow infer \
  --data-csv data/new_samples.csv \
  --run-dir derived/binary \
  --lenient \
  --fill-strategy median \
  --outdir derived/inference_lenient
```

### From Bundle

```bash
classiflow infer \
  --data-csv data/new_samples.csv \
  --bundle model_bundle.zip \
  --outdir derived/inference
```

## Output Structure

```
outdir/
├── predictions.csv           # Main predictions
├── metrics.xlsx              # Metrics workbook (if labels provided)
├── confusion_matrix.png      # Confusion matrix
├── roc_curve.png             # ROC curve
├── pr_curve.png              # Precision-recall curve
└── inference_config.json     # Configuration used
```

## Predictions CSV Format

```csv
sample_id,diagnosis,predicted_label,predicted_proba_Benign,predicted_proba_Malignant,binary_task_score
sample_001,Malignant,Malignant,0.12,0.88,0.88
sample_002,Benign,Benign,0.95,0.05,0.05
...
```

## Feature Alignment

### Strict Mode (Default)

- Requires all training features present
- Fails if any feature missing
- Safest option

### Lenient Mode

- Fills missing features with zeros or medians
- Warns about missing features
- Use when feature sets differ slightly

!!! warning "Lenient Mode Caveat"
    Missing features may degrade performance. Use strict mode when possible.

## See Also

- [train-binary](train-binary.md) - Train binary classifier
- [bundle](bundle.md) - Create model bundles
