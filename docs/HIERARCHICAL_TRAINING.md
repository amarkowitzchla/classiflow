# Hierarchical Training with Patient-Level Stratification

This guide explains how to use the hierarchical nested cross-validation training mode with patient-level stratification and GPU acceleration.

## Features

### Core Capabilities
- **Single-Label Classification**: Standard multiclass classification
- **Hierarchical Two-Level Classification**: L1 → branch-specific L2 models
- **Patient-Level Stratification**: Prevents data leakage by splitting at patient/slide level
- **GPU Acceleration**: Automatic support for CUDA and MPS (Apple Silicon)
- **Early Stopping**: Prevents overfitting with validation-based stopping
- **Hyperparameter Tuning**: Grid search over MLP architectures in inner CV

### Device Support
- **CPU**: Always available
- **CUDA**: NVIDIA GPU acceleration (auto-detected)
- **MPS**: Apple Silicon GPU acceleration (auto-detected)
- **Auto**: Automatically selects best available device (MPS → CUDA → CPU)

## Quick Start

### Single-Label Classification

Train a standard multiclass classifier:

```bash
classiflow train-hierarchical \
    --data-csv data/features.csv \
    --patient-col patient_id \
    --label-l1 diagnosis \
    --device auto \
    --outer-folds 3 \
    --inner-splits 3 \
    --outdir results/single_label
```

### Hierarchical Two-Level Classification

Train hierarchical L1 → L2 classifiers:

```bash
classiflow train-hierarchical \
    --data-csv data/features.csv \
    --patient-col patient_id \
    --label-l1 tumor_type \
    --label-l2 subtype \
    --device cuda \
    --outer-folds 5 \
    --inner-splits 3 \
    --use-smote \
    --outdir results/hierarchical
```

### Hierarchical with L2 Class Filtering

Train only on specific L2 subtypes:

```bash
classiflow train-hierarchical \
    --data-csv data/features.csv \
    --patient-col svs_id \
    --label-l1 primary_diagnosis \
    --label-l2 molecular_subtype \
    --l2-classes subtype_A subtype_B subtype_C \
    --min-l2-classes-per-branch 2 \
    --device mps \
    --outdir results/filtered
```

## Data Format

### CSV Requirements

Your CSV must contain:
1. **Patient/Slide ID column**: For stratification (e.g., `patient_id`, `svs_id`)
2. **Label columns**:
   - `label-l1`: Primary label (required)
   - `label-l2`: Secondary label (optional, enables hierarchical mode)
3. **Feature columns**: All numeric columns (auto-detected)

### Example Data Structure

```csv
patient_id,diagnosis,subtype,feature_1,feature_2,...,feature_N
P001,TypeA,SubtypeX,0.123,0.456,...,0.789
P001,TypeA,SubtypeX,0.234,0.567,...,0.890
P002,TypeB,SubtypeY,0.345,0.678,...,0.901
P002,TypeB,SubtypeY,0.456,0.789,...,0.012
P003,TypeA,SubtypeZ,0.567,0.890,...,0.123
...
```

**Important Notes:**
- Multiple rows per patient are allowed (e.g., multiple tiles per slide)
- Stratification uses majority L1 label per patient
- Missing L2 values (NaN) are allowed - those samples won't be used for L2 training
- All numeric columns (excluding ID and labels) are used as features

## Patient-Level Stratification

### Why It Matters

Standard cross-validation can leak information when you have multiple samples per patient:
- Training set: Patient P001, Tile 1
- Validation set: Patient P001, Tile 2

This inflates performance metrics because the model has seen data from the same patient.

### How It Works

1. **Grouping**: Group all samples by patient ID
2. **Label Assignment**: Assign majority L1 label to each patient
3. **Stratified Split**: Split patients (not samples) into folds
4. **Sample Assignment**: All samples from a patient go to the same fold

This ensures **zero data leakage** between train and validation sets.

## Hierarchical Classification

### Architecture

```
Input Features
      ↓
Level-1 Classifier (L1)
      ↓
   Prediction
      ↓
   ┌─────┴─────┐
   │           │
Branch A    Branch B
(L2-A)      (L2-B)
```

### Training Process

1. **Level-1**: Train multiclass classifier on all data using L1 labels
2. **Level-2**: For each L1 class, train branch-specific L2 classifier
3. **Filtering**:
   - Only train L2 branch if ≥ `min_l2_classes_per_branch` classes
   - Only use samples with non-NA L2 labels
   - Optionally filter by `l2_classes` parameter

### Inference Pipeline

```python
# 1. Predict L1
l1_prediction = model_l1.predict(features)

# 2. Route to L2 branch
if l1_prediction == "TypeA":
    l2_prediction = model_l2_A.predict(features)
else:
    l2_prediction = model_l2_B.predict(features)

# 3. Combined prediction
final_prediction = f"{l1_prediction}::{l2_prediction}"
```

## PyTorch MLP Architecture

### Default Configuration

```python
TorchMLP(
    input_dim=n_features,
    hidden_dims=[128],           # Single hidden layer
    num_classes=n_classes,
    dropout=0.3
)
```

### Hyperparameter Search Grid

The inner CV searches over:

| Parameter | Options |
|-----------|---------|
| Hidden Dims | `[128]`, `[256]`, `[128, 64]`, `[128, 128]` |
| Learning Rate | `1e-3`, `5e-4` |
| Dropout | `0.2`, `0.3`, `0.4` |
| Epochs | Fixed (with early stopping) |

### Training Details

- **Optimizer**: AdamW with weight decay (1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss**: CrossEntropyLoss
- **Gradient Clipping**: Norm clipping at 1.0
- **Batch Normalization**: Applied after each linear layer
- **Early Stopping**: Patience=10 epochs on validation loss

## Command-Line Arguments

### Required
- `--data-csv`: Path to CSV with features and labels
- `--label-l1`: Column name for Level-1 labels

### Data
- `--patient-col`: Patient/slide ID column (default: `svs_id`)
- `--label-l2`: Level-2 label column (enables hierarchical mode)
- `--l2-classes`: Whitelist of L2 classes to include
- `--min-l2-classes-per-branch`: Min L2 classes to train branch (default: 2)

### Output
- `--outdir`: Output directory (default: `derived_hierarchical`)
- `--output-format`: Metrics format - `xlsx` or `csv` (default: `xlsx`)

### Cross-Validation
- `--outer-folds`: Number of outer CV folds (default: 3)
- `--inner-splits`: Number of inner CV splits (default: 3)
- `--random-state`: Random seed (default: 42)

### PyTorch MLP
- `--device`: Device - `auto`, `cpu`, `cuda`, `mps` (default: `auto`)
- `--mlp-epochs`: Max epochs (default: 100)
- `--mlp-batch-size`: Batch size (default: 256)
- `--mlp-hidden`: Base hidden dimension (default: 128)
- `--mlp-dropout`: Dropout rate (default: 0.3)
- `--early-stopping-patience`: Early stopping patience (default: 10)

### SMOTE
- `--use-smote`: Enable SMOTE for class balancing
- `--smote-k-neighbors`: Number of neighbors (default: 5)

### Logging
- `--verbose`: Verbosity level (0=minimal, 1=standard, 2=detailed)

## Output Structure

```
results/
├── training_config.json              # Configuration used
├── metrics_inner_cv.xlsx             # Inner CV hyperparameter results
├── metrics_outer_eval.xlsx           # Outer CV evaluation per fold
├── metrics_summary.xlsx              # Aggregated mean ± std
├── fold1/
│   ├── patient_split_fold1.csv       # Patient assignments
│   ├── scaler.joblib                 # Feature scaler
│   ├── label_encoder_l1.joblib       # L1 label encoder
│   ├── model_level1_fold1.pt         # L1 model weights
│   ├── model_config_l1_fold1.json    # L1 model config
│   ├── label_encoder_l2_TypeA.joblib # L2 encoders per branch
│   ├── model_level2_TypeA_fold1.pt   # L2 models per branch
│   └── model_config_l2_TypeA_fold1.json
├── fold2/
│   └── ...
└── fold3/
    └── ...
```

## Performance Metrics

### Single-Label Evaluation

For each fold, reports:
- **Accuracy**: Overall accuracy
- **Balanced Accuracy**: Average of per-class accuracies
- **F1 Macro**: Macro-averaged F1 score

### Hierarchical Evaluation

Reports all of the above, plus:
- **Per-Branch Metrics**: Separate metrics for each L2 branch
- **Pipeline Metrics**: End-to-end L1→L2 combined predictions

### Summary Statistics

`metrics_summary.xlsx` contains:
- Mean ± std across all folds
- Separate rows for L1, each L2 branch, and pipeline

## GPU Acceleration

### Checking Device

```bash
# Check what device will be used
python -c "from classiflow.models.torch_mlp import resolve_device; print(resolve_device('auto'))"
```

### Expected Speedup

Compared to CPU:
- **CUDA (NVIDIA)**: 5-20x faster depending on GPU
- **MPS (Apple Silicon)**: 2-5x faster on M1/M2/M3

### Memory Requirements

Rule of thumb:
- **Small datasets** (<10K samples): 2-4 GB VRAM
- **Medium datasets** (10K-100K): 4-8 GB VRAM
- **Large datasets** (>100K): 8-16 GB VRAM

Adjust `--mlp-batch-size` if you run out of memory:
```bash
# Reduce batch size for limited VRAM
--mlp-batch-size 128  # Instead of default 256
```

## Advanced Examples

### Custom Hyperparameter Search

Modify the grid in `src/classiflow/training/hierarchical_cv.py`:

```python
def get_hyperparam_candidates(base_hidden: int, base_epochs: int) -> List[Dict]:
    return [
        {"hidden_dims": [base_hidden], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
        {"hidden_dims": [base_hidden * 2], "lr": 1e-3, "epochs": base_epochs, "dropout": 0.3},
        # Add more configurations here
    ]
```

### Large Dataset Training

For datasets >100K samples:

```bash
classiflow train-hierarchical \
    --data-csv large_data.csv \
    --patient-col patient_id \
    --label-l1 diagnosis \
    --device cuda \
    --mlp-batch-size 512 \
    --mlp-epochs 50 \
    --early-stopping-patience 5 \
    --outer-folds 3 \
    --inner-splits 2 \
    --verbose 1
```

### Small Dataset (High Variance)

For small datasets (<1K samples):

```bash
classiflow train-hierarchical \
    --data-csv small_data.csv \
    --patient-col patient_id \
    --label-l1 diagnosis \
    --device cpu \
    --mlp-batch-size 64 \
    --mlp-epochs 200 \
    --early-stopping-patience 20 \
    --outer-folds 5 \
    --inner-splits 5 \
    --use-smote \
    --verbose 2
```

## Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
--mlp-batch-size 128  # or 64
```

### Error: "MPS not available"

**Solution**: Use CUDA or CPU
```bash
--device cuda  # or --device cpu
```

### Warning: "Skipping branch: only N L2 classes"

**Cause**: Branch has fewer than `min_l2_classes_per_branch` classes

**Solution**:
- Reduce `--min-l2-classes-per-branch 1` (allows binary branches)
- Or accept that this branch won't have L2 predictions

### Error: "No numeric feature columns found"

**Cause**: All columns are non-numeric or excluded

**Solution**: Check your CSV has numeric columns besides ID and labels

### Poor Performance

**Try**:
1. Increase epochs: `--mlp-epochs 200`
2. Enable SMOTE: `--use-smote`
3. Increase hidden size: `--mlp-hidden 256`
4. More inner CV: `--inner-splits 5`
5. Check for class imbalance in your data

## Comparison with Other Training Modes

| Feature | Binary | Meta | Hierarchical |
|---------|--------|------|--------------|
| Task Type | Binary | Multiclass (OvR) | Multiclass + Hierarchical |
| Patient Stratification | ❌ | ❌ | ✅ |
| GPU Support | ❌ | ❌ | ✅ |
| Early Stopping | ❌ | ❌ | ✅ |
| Hierarchical L1→L2 | ❌ | ❌ | ✅ |
| Model Type | Sklearn | Sklearn | PyTorch MLP |
| Best For | Simple binary | OvR multiclass | Complex multiclass + hierarchies |

## Python API

You can also use the training function directly in Python:

```python
from pathlib import Path
from classiflow.config import HierarchicalConfig
from classiflow.training.hierarchical_cv import train_hierarchical

config = HierarchicalConfig(
    data_csv=Path("data/features.csv"),
    patient_col="patient_id",
    label_l1="diagnosis",
    label_l2="subtype",
    device="auto",
    outer_folds=5,
    inner_splits=3,
    use_smote=True,
    outdir=Path("results"),
    verbose=2,
)

results = train_hierarchical(config)
print(results)
```

## References

### Key Differences from Original Script

The package implementation differs from `scripts/hierarchical_nested_cv_trainer.py`:

1. **Integrated into package**: Proper module structure with imports
2. **CLI integration**: Full Typer CLI with all options
3. **Config-based**: Uses HierarchicalConfig dataclass
4. **Simplified plotting**: Removed ROC/PR curves (can be added via separate utility)
5. **Cleaner logging**: Uses standard Python logging
6. **Better error handling**: Graceful failures with informative messages

### Future Enhancements

Potential additions (not yet implemented):
- ROC and PR curve plotting per fold
- Averaged curves across folds
- Inference CLI command for trained models
- Feature importance analysis
- Confusion matrices
- Per-class performance breakdown

---

For questions or issues, please open an issue on GitHub or contact:
- **Email**: alexmarkowitz@ucla.edu
- **GitHub**: https://github.com/alexmarkowitz/classiflow
