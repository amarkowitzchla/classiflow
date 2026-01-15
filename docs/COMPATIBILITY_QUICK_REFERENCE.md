# Data Compatibility Quick Reference

## 📋 Quick Check Commands

### Meta-Classifier Mode
```bash
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis
```

### Hierarchical Mode (Single-Level)
```bash
classiflow check-compatibility \
  --data-csv data.csv \
  --mode hierarchical \
  --label-col diagnosis \
  --patient-col patient_id
```

### Hierarchical Mode (Two-Level)
```bash
classiflow check-compatibility \
  --data-csv data.csv \
  --mode hierarchical \
  --label-col tumor_type \
  --label-l2 subtype \
  --patient-col patient_id
```

---

## 📊 Minimum Requirements

### Meta-Classifier Mode
| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Samples | 10 | 100+ |
| Features | 1 numeric | 10+ |
| Classes | 3 | 3-10 |
| Samples/class | 2 | 30+ |

### Hierarchical Mode
| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Samples | 10 | 100+ |
| Patients | outer_folds (default: 3) | 30+ |
| Features | 1 numeric | 10+ |
| L1 Classes | 2 | 2-5 |
| L2 Classes/branch | 2 | 2-5 |

---

## 🚨 Common Errors & Quick Fixes

### Error: Too Few Classes (< 3)
**Problem**: Meta mode needs 3+ classes
**Fix**:
- Use `classiflow train-binary` for 2 classes
- Or collect more classes

### Error: No Numeric Features
**Problem**: All columns are text/categorical
**Fix**:
```python
# Convert to numeric before training
df['category'] = pd.Categorical(df['category']).codes
```

### Error: Single-Sample Classes
**Problem**: Class has only 1 sample
**Fix**:
```bash
# Exclude rare classes
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis \
  --classes ClassA ClassB ClassC  # Exclude rare class
```

### Error: Too Few Patients for CV
**Problem**: Not enough patients for k-fold CV
**Fix**:
```bash
# Reduce outer_folds
classiflow train-hierarchical \
  --data-csv data.csv \
  --label-l1 diagnosis \
  --outer-folds 2  # Reduced from 3
```

### Error: Missing Label Column
**Problem**: Column name doesn't exist
**Fix**:
```bash
# Check column names
head -1 data.csv

# Use correct column name
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col CORRECT_COLUMN_NAME
```

---

## ⚠️  Common Warnings & Solutions

### Warning: Severe Class Imbalance
**Solution**: Use SMOTE
```bash
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis \
  --smote both  # Compare with/without SMOTE
```

### Warning: Missing Values in Features
**Solution**: Impute before training
```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('data.csv')
imputer = SimpleImputer(strategy='median')
feature_cols = df.select_dtypes(include=[np.number]).columns
df[feature_cols] = imputer.fit_transform(df[feature_cols])
df.to_csv('data_imputed.csv', index=False)
```

### Warning: Constant Features
**Solution**: Nothing needed - automatically removed during training

### Warning: Infinite Values
**Solution**: Replace with NaN then impute
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median())
df.to_csv('data_fixed.csv', index=False)
```

---

## 🐍 Python API Quick Reference

### Basic Check
```python
from classiflow.config import MetaConfig
from classiflow.io.compatibility import assess_data_compatibility

config = MetaConfig(data_csv="data.csv", label_col="diagnosis")
result = assess_data_compatibility(config, return_details=True)

if result.is_compatible:
    print("✓ Ready to train!")
else:
    print("✗ Issues found:")
    for error in result.errors:
        print(f"  - {error}")
```

### Print Full Report
```python
from classiflow.io.compatibility import print_compatibility_report

config = MetaConfig(data_csv="data.csv", label_col="diagnosis")
result = print_compatibility_report(config)
```

### Access Data Summary
```python
result = assess_data_compatibility(config, return_details=True)

print(f"Samples: {result.data_summary['n_samples']}")
print(f"Features: {result.data_summary['n_features']}")
print(f"Classes: {result.data_summary['n_classes']}")

# Class distribution
for cls, count in result.data_summary['class_distribution'].items():
    print(f"  {cls}: {count} samples")
```

---

## 📁 Required CSV Format

### Meta-Classifier
```csv
feature_1,feature_2,feature_3,diagnosis
0.5,1.2,0.8,ClassA
1.2,0.9,0.3,ClassB
0.1,1.5,0.9,ClassC
```

### Hierarchical (Single-Level)
```csv
patient_id,feature_1,feature_2,diagnosis
patient_A,0.5,1.2,TypeA
patient_A,0.6,1.1,TypeA
patient_B,1.2,0.9,TypeB
```

### Hierarchical (Two-Level)
```csv
patient_id,feature_1,feature_2,tumor_type,subtype
patient_A,0.5,1.2,TypeA,SubA1
patient_A,0.6,1.1,TypeA,SubA2
patient_B,1.2,0.9,TypeB,SubB1
```

---

## 🎯 Decision Tree: Which Mode?

```
Do you have 2 classes?
├─ YES → Use: classiflow train-binary
└─ NO (3+ classes)
   └─ Do you have repeated measures per patient/subject?
      ├─ NO → Use: classiflow train-meta
      └─ YES
         └─ Do you have hierarchical labels (L1 → L2)?
            ├─ NO → Use: classiflow train-hierarchical (single-level)
            └─ YES → Use: classiflow train-hierarchical (two-level)
```

---

## 💡 Best Practices

1. **Always check first**: Run `check-compatibility` before training
2. **Address warnings**: Fix non-critical issues for best results
3. **Use SMOTE**: For imbalanced datasets, try `--smote both`
4. **Specify features**: Use `feature_cols` for reproducibility
5. **Clean data**: Remove inf, impute NaN before training
6. **More data**: More samples per class = better performance

---

## 📚 Full Documentation

- **Detailed Guide**: [DATA_COMPATIBILITY_GUIDE.md](DATA_COMPATIBILITY_GUIDE.md)
- **Examples**: [example_check_compatibility.py](example_check_compatibility.py)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 🆘 Getting Help

```bash
# CLI help
classiflow check-compatibility --help
classiflow train-meta --help
classiflow train-hierarchical --help

# Run examples
python example_check_compatibility.py
```

For issues not covered here, see [DATA_COMPATIBILITY_GUIDE.md](DATA_COMPATIBILITY_GUIDE.md) for comprehensive documentation.
