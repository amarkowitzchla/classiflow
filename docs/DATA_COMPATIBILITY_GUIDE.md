# Data Compatibility Guide

This guide explains how to check if your data is compatible with classiflow's training modes and how to fix common issues.

## Quick Start

Before training, check if your data is compatible:

```bash
# Check for meta-classifier mode
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis

# Check for hierarchical mode
classiflow check-compatibility \
  --data-csv data.csv \
  --mode hierarchical \
  --label-col tumor_type \
  --label-l2 subtype
```

The compatibility check will:
- ✓ Verify your data meets all requirements
- ⚠️  Warn about potential issues
- ✗ Report errors that must be fixed
- 💡 Suggest how to fix incompatibilities

## Table of Contents

1. [Meta-Classifier Requirements](#meta-classifier-requirements)
2. [Hierarchical Mode Requirements](#hierarchical-mode-requirements)
3. [Common Issues and Solutions](#common-issues-and-solutions)
4. [Using the Compatibility Check](#using-the-compatibility-check)
5. [Python API](#python-api)

---

## Meta-Classifier Requirements

The **train-meta** mode is for multiclass classification problems where you want to combine binary classifiers.

### Minimum Requirements

| Requirement | Value | Why |
|------------|-------|-----|
| Samples | ≥ 10 | Minimum for cross-validation |
| Features | ≥ 1 numeric column | Need features to train on |
| Classes | ≥ 3 unique classes | Meta-classifier requires multiclass problem |
| Samples per class | ≥ 2 | Minimum for stratified splits |

### Data Format

**CSV Structure:**
```csv
feature_1,feature_2,...,feature_N,diagnosis
0.5,1.2,...,0.8,ClassA
1.2,0.9,...,0.3,ClassB
0.1,1.5,...,0.9,ClassA
```

**Requirements:**
- One label column (string or numeric)
- Multiple numeric feature columns
- No missing values in labels (missing label rows will be dropped)
- Features should be numeric (auto-detected by default)

### Example: Valid Meta Data

```csv
gene_1,gene_2,gene_3,diagnosis
0.5,1.2,0.8,Subtype_A
1.2,0.9,0.3,Subtype_B
0.1,1.5,0.9,Subtype_C
0.9,1.1,0.7,Subtype_A
1.3,0.8,0.4,Subtype_B
0.2,1.6,1.0,Subtype_C
```

### What Gets Checked

✓ **File Exists**: CSV file is readable
✓ **Label Column**: Specified column exists
✓ **Numeric Features**: At least one numeric column
✓ **Minimum Samples**: ≥ 10 samples total
✓ **Minimum Classes**: ≥ 3 unique classes
✓ **Class Balance**: Each class has ≥ 2 samples
⚠️ **Class Imbalance**: Warns if ratio > 10:1
⚠️ **Missing Values**: Warns about NaN in features
⚠️ **Constant Features**: Warns about zero-variance features
✗ **Infinite Values**: Rejects if any feature has inf/-inf

---

## Hierarchical Mode Requirements

The **train-hierarchical** mode supports:
- Single-level classification (L1 only)
- Hierarchical two-level classification (L1 → L2 per branch)
- Patient-level stratification (prevents data leakage)

### Minimum Requirements

| Requirement | Value | Why |
|------------|-------|-----|
| Samples | ≥ 10 | Minimum for cross-validation |
| Patients | ≥ outer_folds | Need enough for patient-level CV |
| Features | ≥ 1 numeric column | Need features to train on |
| L1 Classes | ≥ 2 unique classes | Binary or multiclass |
| Samples per L1 class | ≥ 2 | Minimum for stratified splits |
| L2 Classes per branch | ≥ 2 (if hierarchical) | Minimum for meaningful L2 classification |

### Data Format - Single Level

**CSV Structure:**
```csv
svs_id,feature_1,feature_2,...,feature_N,diagnosis
patient_A,0.5,1.2,...,0.8,TypeA
patient_A,0.6,1.1,...,0.7,TypeA
patient_B,1.2,0.9,...,0.3,TypeB
patient_B,1.1,0.8,...,0.4,TypeB
```

**Requirements:**
- Patient/sample ID column (for stratification)
- L1 label column
- Multiple numeric feature columns

### Data Format - Hierarchical (Two-Level)

**CSV Structure:**
```csv
svs_id,feature_1,feature_2,...,feature_N,tumor_type,subtype
patient_A,0.5,1.2,...,0.8,GliomaA,SubA1
patient_A,0.6,1.1,...,0.7,GliomaA,SubA2
patient_B,1.2,0.9,...,0.3,GliomaB,SubB1
patient_B,1.1,0.8,...,0.4,GliomaB,SubB1
patient_C,2.1,0.5,...,0.2,GliomaA,
```

**Requirements:**
- Patient/sample ID column
- L1 label column (tumor_type)
- L2 label column (subtype)
- L2 can have missing values for some samples
- Each L1 branch should have ≥ 2 L2 classes (configurable)

### What Gets Checked

✓ **File Exists**: CSV file is readable
✓ **Patient Column**: Specified column exists
✓ **L1 Label Column**: Specified column exists
✓ **L2 Label Column**: (If hierarchical) specified column exists
✓ **Numeric Features**: At least one numeric column
✓ **Minimum Samples**: ≥ 10 samples total
✓ **Minimum Patients**: ≥ outer_folds (e.g., 3 patients for 3-fold CV)
✓ **L1 Classes**: ≥ 2 unique classes
✓ **L2 per Branch**: (If hierarchical) each branch has enough L2 classes
⚠️ **Few Patients**: Warns if < 30 patients
⚠️ **Missing L2**: Warns if some samples missing L2 labels
⚠️ **Insufficient L2 Branches**: Warns if branches don't meet min_l2_classes_per_branch

---

## Common Issues and Solutions

### Issue 1: Too Few Classes for Meta Mode

**Error:**
```
Too few classes: 2 (meta-classifier requires at least 3)
```

**Solution:**
- Meta mode is for **multiclass** problems (3+ classes)
- For **binary** classification, use `classiflow train-binary` instead:

```bash
classiflow train-binary \
  --data-csv data.csv \
  --label-col diagnosis \
  --pos-label ClassA
```

### Issue 2: Single-Sample Classes

**Error:**
```
Classes with < 2 samples: ['RareClass']
```

**Solutions:**
1. Remove rare classes from analysis:
   ```bash
   classiflow train-meta \
     --data-csv data.csv \
     --label-col diagnosis \
     --classes ClassA ClassB ClassC  # Exclude RareClass
   ```

2. Collect more samples for rare classes

3. Combine rare classes into "Other" category in your CSV

### Issue 3: No Numeric Features

**Error:**
```
No numeric feature columns found
```

**Solutions:**
1. Ensure your features are numeric (not text/categorical)
2. Convert categorical features to numeric before training
3. If columns should be numeric, check CSV formatting
4. Explicitly specify feature columns:
   ```python
   config = MetaConfig(
       data_csv="data.csv",
       label_col="diagnosis",
       feature_cols=["gene_1", "gene_2", "gene_3"]
   )
   ```

### Issue 4: Too Few Patients for CV

**Error:**
```
Too few patients (2) for 3-fold CV
```

**Solutions:**
1. Reduce `outer_folds`:
   ```bash
   classiflow train-hierarchical \
     --data-csv data.csv \
     --label-l1 diagnosis \
     --outer-folds 2  # Reduced from 3
   ```

2. Collect more patient data

### Issue 5: Severe Class Imbalance

**Warning:**
```
Severe class imbalance detected (ratio: 50.0:1)
```

**Solutions:**
1. Use SMOTE to balance classes:
   ```bash
   classiflow train-meta \
     --data-csv data.csv \
     --label-col diagnosis \
     --smote both  # Try with and without SMOTE
   ```

2. Collect more samples from minority classes

3. Use class weights (automatic in LogisticRegression)

### Issue 6: Missing Values in Features

**Warning:**
```
5 features have missing values
```

**Solutions:**
1. Impute missing values before training:
   ```python
   import pandas as pd
   from sklearn.impute import SimpleImputer

   df = pd.read_csv("data.csv")
   feature_cols = df.select_dtypes(include=[np.number]).columns
   imputer = SimpleImputer(strategy='median')
   df[feature_cols] = imputer.fit_transform(df[feature_cols])
   df.to_csv("data_imputed.csv", index=False)
   ```

2. Remove features with too many missing values

3. Use advanced imputation (KNN, iterative)

### Issue 7: Infinite Values

**Error:**
```
1 features contain infinite values
```

**Solution:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# Replace infinite with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Then impute
df = df.fillna(df.median())

df.to_csv("data_fixed.csv", index=False)
```

### Issue 8: Constant Features

**Warning:**
```
10 constant features (zero variance)
```

**Solution:**
- Don't worry! These are automatically removed during training
- Or remove them manually beforehand:

```python
import pandas as pd

df = pd.read_csv("data.csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns
constant_cols = [col for col in numeric_cols if df[col].std() == 0]

df = df.drop(columns=constant_cols)
df.to_csv("data_cleaned.csv", index=False)
```

---

## Using the Compatibility Check

### CLI Command

The `check-compatibility` command validates your data before training:

```bash
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis
```

**Options:**

| Option | Description | Required |
|--------|-------------|----------|
| `--data-csv` | Path to CSV file | Yes |
| `--mode` | Training mode: `meta` or `hierarchical` | Yes |
| `--label-col` | Label column name (or L1 for hierarchical) | Yes |
| `--label-l2` | L2 label column (hierarchical only) | No |
| `--patient-col` | Patient ID column (hierarchical, default: svs_id) | No |
| `--classes` | Subset of classes to include (meta only) | No |
| `--outer-folds` | Number of CV folds (default: 3) | No |

### Example Output

**Compatible Data:**
```
============================================================
  DATA COMPATIBILITY ASSESSMENT - META MODE
============================================================

Status: ✓ COMPATIBLE

Data Summary:
  • Samples: 150
  • Features: 30
  • Classes: 4

  Class Distribution:
    - ClassA: 50 samples
    - ClassB: 40 samples
    - ClassC: 35 samples
    - ClassD: 25 samples

────────────────────────────────────────────────────────────
WARNINGS:
  1. 5 features have missing values

────────────────────────────────────────────────────────────
SUGGESTIONS:
  1. Consider imputing missing values before training

============================================================

✓ Data is compatible but has warnings (see above)
```

**Incompatible Data:**
```
============================================================
  DATA COMPATIBILITY ASSESSMENT - META MODE
============================================================

Status: ✗ INCOMPATIBLE

Data Summary:
  • Samples: 50
  • Features: 10
  • Classes: 2

  Class Distribution:
    - ClassA: 30 samples
    - ClassB: 20 samples

────────────────────────────────────────────────────────────
ERRORS:
  1. Too few classes: 2 (meta-classifier requires at least 3)

────────────────────────────────────────────────────────────
SUGGESTIONS:
  1. Meta-classifier is for multiclass problems (3+ classes)
  2. For binary classification, use 'classiflow train' instead

============================================================
```

### Automatic Check During Training

Both `train-meta` and `train-hierarchical` now run compatibility checks automatically:

```bash
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis

# Output:
# Checking data compatibility...
#
# ============================================================
#   DATA COMPATIBILITY ASSESSMENT - META MODE
# ============================================================
# ...
```

If data is incompatible, training will abort with suggestions.

If there are warnings, you'll be prompted:
```
⚠️  Proceeding with warnings (see above)
Continue with training? [Y/n]:
```

---

## Python API

### Basic Usage

```python
from pathlib import Path
from classiflow.config import MetaConfig, HierarchicalConfig
from classiflow.io.compatibility import assess_data_compatibility

# Create config
config = MetaConfig(
    data_csv=Path("data.csv"),
    label_col="diagnosis"
)

# Check compatibility
result = assess_data_compatibility(config, return_details=True)

# Check result
if result.is_compatible:
    print("✓ Data is compatible!")
else:
    print("✗ Data is NOT compatible")
    for error in result.errors:
        print(f"  - {error}")

    print("\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")
```

### CompatibilityResult Object

The `assess_data_compatibility` function returns a `CompatibilityResult` object:

```python
@dataclass
class CompatibilityResult:
    is_compatible: bool              # True if data is compatible
    mode: str                        # "meta" or "hierarchical"
    warnings: List[str]              # Non-critical issues
    errors: List[str]                # Critical issues (prevent training)
    suggestions: List[str]           # Actionable fixes
    data_summary: Dict[str, Any]     # Data statistics
    schema: DataSchema               # Pydantic schema object
```

### Detailed Example

```python
from classiflow.io.compatibility import print_compatibility_report

# Convenience function that assesses and prints
config = MetaConfig(
    data_csv="data.csv",
    label_col="diagnosis",
    classes=["ClassA", "ClassB", "ClassC"]
)

result = print_compatibility_report(config)

# Access details
print(f"Samples: {result.data_summary['n_samples']}")
print(f"Features: {result.data_summary['n_features']}")
print(f"Classes: {result.data_summary['n_classes']}")

# Access class distribution
for cls, count in result.data_summary['class_distribution'].items():
    print(f"  {cls}: {count} samples")

# Convert to dict (for JSON export)
result_dict = result.to_dict()
```

### Hierarchical Mode Example

```python
from classiflow.config import HierarchicalConfig
from classiflow.io.compatibility import assess_data_compatibility

config = HierarchicalConfig(
    data_csv="data.csv",
    patient_col="patient_id",
    label_l1="tumor_type",
    label_l2="subtype",
    min_l2_classes_per_branch=2,
    outer_folds=3
)

result = assess_data_compatibility(config, return_details=True)

if result.data_summary['hierarchical']:
    print("Hierarchical mode enabled")
    print(f"L1 classes: {result.data_summary['l1_classes']}")
    print(f"L2 branches: {result.data_summary['l2_classes_per_branch']}")
```

---

## Best Practices

### 1. Always Check Before Training

Run compatibility check before starting long training jobs:

```bash
# Check first
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis

# If compatible, train
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis
```

### 2. Address Warnings

Even if data is compatible, address warnings for best results:
- Impute missing values
- Remove constant features
- Consider SMOTE for imbalanced classes

### 3. Use Explicit Feature Columns

For reproducibility, specify feature columns explicitly:

```python
config = MetaConfig(
    data_csv="data.csv",
    label_col="diagnosis",
    feature_cols=["gene_1", "gene_2", "gene_3"]  # Explicit
)
```

### 4. Validate CSV Format

Ensure your CSV is properly formatted:
- No extra quotes or delimiters
- Consistent column names
- Numeric features are actually numeric (not "1.23" as text)

### 5. Patient-Level Stratification

For hierarchical mode with repeated measures:
- Use unique patient IDs (not sample IDs)
- Ensures no patient appears in both train and test
- Prevents data leakage

---

## FAQ

**Q: How many samples do I need?**
A: Minimum 10, but recommend 30+ per class for robust training. More is better.

**Q: Can I use non-numeric features?**
A: No, classiflow requires numeric features. Convert categorical features to numeric (one-hot encoding, label encoding) before training.

**Q: What if I have missing values?**
A: The system will warn you. Impute them before training for best results, or they'll be handled during preprocessing with potential performance loss.

**Q: How do I handle class imbalance?**
A: Use `--smote both` to compare performance with and without SMOTE.

**Q: Can I skip the compatibility check?**
A: No, it runs automatically before training. This prevents wasted time on incompatible data.

**Q: What's the difference between meta and hierarchical modes?**
A:
- **Meta mode**: Multiclass via binary classifiers (3+ classes, sample-level CV)
- **Hierarchical mode**: Two-level classification with patient-level CV (supports repeated measures)

---

## Getting Help

If you encounter issues not covered here:

1. Check the error messages - they contain specific guidance
2. Review the suggestions in the compatibility report
3. See the main README for general documentation
4. File an issue on GitHub with your compatibility report output

## Next Steps

- [Quick Start Guide](QUICK_START_CUSTOM_TASKS.md)
- [Custom Tasks Guide](CUSTOM_TASKS_GUIDE.md)
- [Main README](README.md)
