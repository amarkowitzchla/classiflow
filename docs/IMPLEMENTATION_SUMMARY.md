# Data Compatibility Assessment Implementation

## Summary

I've developed a comprehensive data compatibility assessment system for classiflow that validates input data before training and provides actionable suggestions when data is incompatible.

## What Was Implemented

### 1. Core Compatibility Module
**File**: [src/classiflow/io/compatibility.py](src/classiflow/io/compatibility.py)

A new module that assesses data compatibility with both training modes:
- **Meta-classifier mode**: Validates multiclass data (3+ classes)
- **Hierarchical mode**: Validates single-level or two-level hierarchical data with patient stratification

**Key Features**:
- ✓ Comprehensive validation of data format, features, and labels
- ⚠️  Detects non-critical issues (class imbalance, missing values, constant features)
- ✗ Reports critical errors that prevent training
- 💡 Provides specific, actionable suggestions for fixing issues
- 📊 Detailed data summaries and statistics

### 2. CLI Integration
**File**: [src/classiflow/cli/main.py](src/classiflow/cli/main.py)

#### New Command: `check-compatibility`
Standalone command to validate data before training:

```bash
# Check meta-classifier compatibility
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis

# Check hierarchical compatibility
classiflow check-compatibility \
  --data-csv data.csv \
  --mode hierarchical \
  --label-col tumor_type \
  --label-l2 subtype
```

#### Automatic Checks in Training Commands
Both `train-meta` and `train-hierarchical` now automatically run compatibility checks:
- Shows detailed assessment before training starts
- Aborts if data is incompatible
- Prompts user if there are warnings

### 3. Comprehensive Test Suite
**File**: [tests/unit/test_compatibility.py](tests/unit/test_compatibility.py)

22 comprehensive unit tests covering:
- ✓ Valid data scenarios for both modes
- ✗ Invalid data scenarios (too few classes, samples, missing columns)
- ⚠️  Warning scenarios (class imbalance, missing values, constant features)
- 🔧 Edge cases (hierarchical L2 branches, patient stratification)

**Test Results**: All 22 tests pass ✓

### 4. Documentation
**File**: [DATA_COMPATIBILITY_GUIDE.md](DATA_COMPATIBILITY_GUIDE.md)

Comprehensive 400+ line guide covering:
- Data format requirements for both modes
- Minimum requirements tables
- Common issues and solutions
- CLI usage examples
- Python API examples
- Best practices and FAQ

### 5. Example Script
**File**: [example_check_compatibility.py](example_check_compatibility.py)

Demonstrates 5 usage scenarios:
1. Compatible meta-classifier data
2. Incompatible data (too few classes)
3. Compatible hierarchical data
4. Data with warnings (class imbalance, missing values)
5. Programmatic access to results

## What Gets Checked

### Meta-Classifier Mode

| Check | Type | Description |
|-------|------|-------------|
| File exists | Error | CSV file is readable |
| Label column exists | Error | Specified column present |
| Numeric features | Error | At least one numeric column |
| Minimum samples | Error | ≥ 10 samples |
| Minimum classes | Error | ≥ 3 classes (multiclass requirement) |
| Samples per class | Error | Each class has ≥ 2 samples |
| Infinite values | Error | No inf/-inf in features |
| Class imbalance | Warning | Warns if ratio > 10:1, suggests SMOTE |
| Missing values | Warning | Warns about NaN in features |
| Constant features | Warning | Warns about zero-variance features |

### Hierarchical Mode

| Check | Type | Description |
|-------|------|-------------|
| File exists | Error | CSV file is readable |
| Patient column exists | Error | ID column present |
| L1 label exists | Error | Level-1 label column present |
| L2 label exists (if hierarchical) | Error | Level-2 label column present |
| Numeric features | Error | At least one numeric column |
| Minimum samples | Error | ≥ 10 samples |
| Minimum patients | Error | ≥ outer_folds patients |
| L1 classes | Error | ≥ 2 classes |
| Samples per L1 class | Error | Each class has ≥ 2 samples |
| L2 classes per branch | Warning | Warns if branches lack min L2 classes |
| Few patients | Warning | Warns if < 30 patients |
| Missing L2 labels | Warning | Warns about samples without L2 |

## Usage Examples

### CLI - Standalone Check

```bash
# Check before training
classiflow check-compatibility \
  --data-csv data.csv \
  --mode meta \
  --label-col diagnosis

# Output shows:
# - Status (compatible/incompatible)
# - Data summary (samples, features, classes)
# - Class distribution
# - Errors (if any)
# - Warnings (if any)
# - Suggestions (if any)
```

### CLI - Automatic During Training

```bash
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis

# Automatically runs compatibility check first
# Shows results before starting training
# Prompts if warnings detected
```

### Python API

```python
from pathlib import Path
from classiflow.config import MetaConfig
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
    print(f"Samples: {result.data_summary['n_samples']}")
    print(f"Features: {result.data_summary['n_features']}")
    print(f"Classes: {result.data_summary['n_classes']}")
else:
    print("✗ Data is NOT compatible")
    for error in result.errors:
        print(f"  - {error}")
    for suggestion in result.suggestions:
        print(f"  → {suggestion}")
```

## Example Output

### Compatible Data
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

============================================================
```

### Incompatible Data
```
============================================================
  DATA COMPATIBILITY ASSESSMENT - META MODE
============================================================

Status: ✗ INCOMPATIBLE

Data Summary:
  • Samples: 50
  • Features: 10
  • Classes: 2

────────────────────────────────────────────────────────────
ERRORS:
  1. Too few classes: 2 (meta-classifier requires at least 3)

────────────────────────────────────────────────────────────
SUGGESTIONS:
  1. Meta-classifier is for multiclass problems (3+ classes)
  2. For binary classification, use 'classiflow train' instead

============================================================
```

## Files Modified/Created

### New Files
- `src/classiflow/io/compatibility.py` - Core compatibility assessment module (600+ lines)
- `tests/unit/test_compatibility.py` - Comprehensive test suite (400+ lines)
- `DATA_COMPATIBILITY_GUIDE.md` - User documentation (700+ lines)
- `example_check_compatibility.py` - Example usage script (200+ lines)
- `IMPLEMENTATION_SUMMARY.md` - This file

### Modified Files
- `src/classiflow/io/__init__.py` - Added exports for compatibility functions
- `src/classiflow/cli/main.py` - Added `check-compatibility` command and automatic checks

## Benefits

1. **Prevents Wasted Time**: Catches data issues before training starts
2. **Clear Error Messages**: Explains exactly what's wrong and how to fix it
3. **Educational**: Helps users understand data requirements
4. **Comprehensive**: Covers all common data issues
5. **Well-Tested**: 22 unit tests ensure reliability
6. **Well-Documented**: 700+ line guide with examples

## Testing

All 22 unit tests pass:

```bash
pytest tests/unit/test_compatibility.py -v
# 22 passed in 2.82s
```

Example script runs successfully:

```bash
python example_check_compatibility.py
# All 5 examples complete successfully
```

## Next Steps

Users can now:

1. **Before Training**: Run `classiflow check-compatibility` to validate data
2. **During Training**: Automatic checks provide immediate feedback
3. **Fix Issues**: Follow suggestions to resolve incompatibilities
4. **Learn**: Read DATA_COMPATIBILITY_GUIDE.md for detailed guidance

## Code Quality

- ✓ Type hints throughout
- ✓ Comprehensive docstrings
- ✓ Pydantic dataclasses for results
- ✓ Well-structured and modular
- ✓ Follows existing codebase patterns
- ✓ Fully tested (22 tests)
- ✓ Extensively documented

## Integration

The compatibility checker integrates seamlessly with existing classiflow infrastructure:
- Uses existing config classes (MetaConfig, HierarchicalConfig)
- Uses existing data loading functions (load_data)
- Uses existing schema validation (DataSchema)
- Follows existing CLI patterns (typer)
- Follows existing testing patterns (pytest)
