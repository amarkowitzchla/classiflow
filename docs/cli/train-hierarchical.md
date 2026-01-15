# train-hierarchical

Train a hierarchical classifier with L1 → L2 routing and patient-level stratification.

## Usage

```bash
classiflow train-hierarchical [OPTIONS]
```

## Required Options

| Option | Description |
|--------|-------------|
| `--data-csv PATH` | Path to CSV file |
| `--label-l1 TEXT` | Level-1 label column (e.g., tumor type) |

## Optional Options

### Data Options

| Option | Default | Description |
|--------|---------|-------------|
| `--patient-col TEXT` | None | Patient/sample ID column for stratification |
| `--label-l2 TEXT` | None | Level-2 label column (enables hierarchical mode) |
| `--l2-classes TEXT...` | All | Subset of L2 classes to include |
| `--min-l2-classes INT` | 2 | Minimum L2 classes per branch |

### Cross-Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outer-folds INT` | 3 | Outer CV folds |
| `--inner-splits INT` | 3 | Inner CV splits |
| `--random-state INT` | 42 | Random seed |

### MLP Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device [auto\|cpu\|cuda\|mps]` | `auto` | Compute device |
| `--mlp-epochs INT` | 100 | Training epochs |
| `--mlp-batch-size INT` | 256 | Batch size |
| `--mlp-hidden INT` | 128 | Hidden layer dimension |
| `--mlp-dropout FLOAT` | 0.3 | Dropout rate |
| `--early-stopping-patience INT` | 10 | Early stopping patience |

### SMOTE Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-smote` | False | Enable SMOTE |
| `--smote-k INT` | 5 | SMOTE k_neighbors |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir PATH` | `derived_hierarchical` | Output directory |
| `--output-format [xlsx\|csv]` | `xlsx` | Output format |
| `--verbose INT` | 1 | Verbosity (0, 1, or 2) |

## Examples

### Flat Multiclass (L1 Only)

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --label-l1 tumor_type \
  --device auto \
  --outdir derived/flat
```

### Two-Level Hierarchical

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --device cuda \
  --use-smote \
  --outdir derived/hierarchical
```

### Full Configuration

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --outer-folds 5 \
  --inner-splits 3 \
  --device auto \
  --mlp-epochs 100 \
  --mlp-batch-size 128 \
  --mlp-hidden 256 \
  --mlp-dropout 0.4 \
  --early-stopping-patience 15 \
  --use-smote \
  --smote-k 5 \
  --outdir derived/hierarchical_full \
  --verbose 2
```

## Patient-Level Stratification

When `--patient-col` is specified:

- Patients are split between train/test (not samples)
- Prevents patient-level data leakage
- Critical for medical imaging with multiple samples per patient

## Output Structure

```
outdir/
├── run.json
├── config.json
├── fold_1/
│   ├── l1_model.pt             # L1 classifier
│   ├── l1_scaler.joblib        # L1 scaler
│   ├── l1_encoder.joblib       # L1 label encoder
│   ├── branch_TypeA/           # Branch-specific L2
│   │   ├── l2_model.pt
│   │   ├── l2_scaler.joblib
│   │   └── l2_encoder.joblib
│   ├── branch_TypeB/
│   │   └── ...
│   └── metrics_outer_eval.xlsx
└── summary_metrics.csv
```

## GPU Selection

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 classiflow train-hierarchical ...

# Use CPU explicitly
classiflow train-hierarchical --device cpu ...
```

## See Also

- [infer](infer.md) - Hierarchical inference
- [Hierarchical Guide](../docs/HIERARCHICAL_TRAINING.md)
