# train-multiclass

Train a direct multiclass classifier with nested cross-validation.

Patient-level stratification (optional): Provide `--patient-col patient_id` to ensure no data leakage by patient across folds. If omitted, sample-level stratification is used.

## Usage

```bash
classiflow train-multiclass [OPTIONS]
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
| `--classes TEXT...` | All classes | Class labels to include (order matters) |
| `--patient-col TEXT` | None | Patient/slide ID column for stratification |

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
| `--smote [off\|on\|both]` | `both` | SMOTE mode |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-iter INT` | 10000 | Max iterations for linear models |
| `--device TEXT` | `auto` | Device: auto, cpu, cuda, mps |
| `--estimator-mode TEXT` | `all` | Estimator selection: all, torch_only, cpu_only |

When `--device` resolves to `mps` or `cuda`, the multiclass registry adds torch-backed
estimators (`torch_linear`, `torch_mlp`) to the nested CV search.
Use `--estimator-mode torch_only` to run only the torch-backed models.

## Examples

### Basic Usage

```bash
classiflow train-multiclass \
  --data-csv data/features.csv \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outdir derived/multiclass
```

### Patient-Level Stratification

```bash
classiflow train-multiclass \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-col subtype \
  --classes TypeA TypeB TypeC \
  --outdir derived/multiclass_patient
```

## See Also

- [train-binary](train-binary.md) - Binary classification with nested CV
- [train-meta](train-meta.md) - Multiclass via binary task ensemble
