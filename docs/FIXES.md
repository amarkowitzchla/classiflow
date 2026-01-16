# Bug Fixes Applied

## Issues Fixed

### 1. Import Error: sklearn/imbalanced-learn Version Conflict

**Error:**
```
ImportError: cannot import name '_safe_tags' from 'sklearn.utils._tags'
```

**Cause:** Version incompatibility between `scikit-learn` and `imbalanced-learn`

**Solutions Provided:**

**Option A: Quick Fix Script** (`fix_imports.sh`)
```bash
./fix_imports.sh
```
This script:
- Uninstalls conflicting versions
- Installs compatible versions (scikit-learn 1.5.2 + imbalanced-learn 0.12.4)
- Reinstalls the package
- Runs tests to verify

**Option B: Mamba/Conda Environment** (Recommended - `setup_env.sh`)
```bash
./setup_env.sh
```
This script:
- Creates a clean conda environment
- Installs Python 3.11 with proper dependency resolution
- Avoids system Python conflicts

**pyproject.toml Update:**
Added version constraints to prevent future conflicts while still accommodating the optional visualization stack:
```toml
dependencies = [
    "scikit-learn>=1.4.0,<2.0.0",
    "imbalanced-learn>=0.12.0,<0.13.0",
    ...
]
```

---

### 2. Pipeline Error: VarianceThreshold Removing All Features

**Error:**
```
ValueError: at least one array or dtype is required
All the 40 fits failed
```

**Cause:** `VarianceThreshold` was removing all features in small CV splits, especially with scaled data and small datasets like iris (4 features)

**Fix:** Removed `VarianceThreshold` from pipeline

**Files Modified:**
- `src/classiflow/training/nested_cv.py`
- `src/classiflow/training/meta.py`

**Before:**
```python
pipe = ImbPipeline([
    ("sampler", sampler),
    ("vth", VarianceThreshold()),  # ← Problematic
    ("scaler", StandardScaler()),
    ("clf", est),
])
```

**After:**
```python
pipe = ImbPipeline([
    ("sampler", sampler),
    ("scaler", StandardScaler()),
    ("clf", est),
])
```

**Rationale:**
- Small CV splits can have very low variance after scaling
- VarianceThreshold with default threshold=0.0 can remove all features
- For small datasets (like iris with 4 features), this is catastrophic
- StandardScaler already handles feature normalization
- Estimators like LogisticRegression and SVM handle collinearity well

---

### 3. Pipeline Error: AdaptiveSMOTE Implements Both fit_resample and transform

**Error:**
```
TypeError: All intermediate steps of the chain should be estimators that implement
fit and transform or fit_resample.
'<classiflow.models.smote.AdaptiveSMOTE object>' implements both
```

**Cause:** `imblearn.Pipeline` requires resamplers to implement ONLY `fit_resample()`, not both `fit_resample()` and `fit()`/`transform()`

**Fix:** Removed `fit()` and `transform()` methods from `AdaptiveSMOTE`

**File Modified:**
- `src/classiflow/models/smote.py`

**Before:**
```python
class AdaptiveSMOTE:
    def fit_resample(self, X, y):
        # ... resampling logic

    def fit(self, X, y):  # ← Problematic
        return self

    def transform(self, X):  # ← Problematic
        return X
```

**After:**
```python
class AdaptiveSMOTE:
    def fit_resample(self, X, y):
        # ... resampling logic
    # fit() and transform() removed
```

**Rationale:**
- `imblearn.Pipeline` uses duck-typing to determine step types
- If both `fit_resample()` and `fit()`/`transform()` exist, it gets confused
- Resamplers should ONLY implement `fit_resample()`
- Standard estimators implement `fit()` and `transform()` or `fit()` and `predict()`

---

### 4. Data Column Name Mismatch

**Error:**
Training failed silently due to wrong label column name

**Cause:** iris_data.csv uses `"Species"` (capitalized) but quickstart.sh used `"species"` (lowercase)

**Fix:** Updated quickstart.sh to use correct column name

**File Modified:**
- `quickstart.sh`

**Before:**
```bash
--label-col species
```

**After:**
```bash
--label-col Species
```

---

## New Files Created

1. **setup_env.sh** - Automated environment setup with mamba/conda
2. **fix_imports.sh** - Quick fix for sklearn/imblearn conflict
3. **ENVIRONMENT.md** - Comprehensive environment setup guide
4. **test_quick.sh** - Minimal test script for rapid validation
5. **FIXES.md** - This document

---

## Testing the Fixes

### Quick Test
```bash
./test_quick.sh
```

### Full Test
```bash
./quickstart.sh
```

### Unit Tests
```bash
pytest -v
```

### Specific Module Tests
```bash
pytest tests/unit/test_smote.py -v
pytest tests/unit/test_tasks.py -v
pytest tests/unit/test_metrics.py -v
```

---

## Verification Checklist

- [x] Import errors resolved
- [x] AdaptiveSMOTE pipeline compatible
- [x] VarianceThreshold removed
- [x] Iris dataset column name corrected
- [x] Version constraints added to pyproject.toml
- [x] Environment setup scripts created
- [x] Documentation updated

---

## Recommended Workflow

1. **First Time Setup:**
   ```bash
   # Option A: Use mamba/conda (recommended)
   ./setup_env.sh

   # Option B: Quick fix on system Python
   ./fix_imports.sh
   ```

2. **Run Tests:**
   ```bash
   pytest -v
   ```

3. **Test Training:**
   ```bash
   ./test_quick.sh
   ```

4. **Full Quickstart:**
   ```bash
   ./quickstart.sh
   ```

---

## Prevention for Future

### For Developers:

1. **Always test with small datasets** (like iris) to catch edge cases
2. **Use imblearn.Pipeline** correctly:
   - Resamplers: implement only `fit_resample()`
   - Transformers: implement `fit()` and `transform()`
   - Estimators: implement `fit()` and `predict()`
3. **Pin dependency versions** in pyproject.toml
4. **Test in clean environments** (mamba/conda) before release

### For Users:

1. **Use mamba/conda** for ML projects (better dependency resolution)
2. **Create isolated environments** per project
3. **Check column names** in your CSV files (case-sensitive!)
4. **Run tests** after installation to verify setup

---

## Contact

If you encounter additional issues:
- **GitHub Issues:** https://github.com/alexmarkowitz/classiflow/issues
- **Email:** alexmarkowitz@ucla.edu

---

## Change Log

### 2026-01-13 (Hierarchical Training Update)
- **Added hierarchical nested CV training** with patient-level stratification
- **Added PyTorch MLP** with CUDA/MPS (Apple Silicon) GPU support
- **Added HierarchicalConfig** dataclass for hierarchical training configuration
- **Added train-hierarchical CLI command** with full option support
- **Created HIERARCHICAL_TRAINING.md** comprehensive guide
- Updated dependencies: added `torch>=2.0.0,<2.6.0`
- Enhanced SMOTE module with standalone `apply_smote()` function
- Updated README with hierarchical training examples

**New Files:**
- `src/classiflow/models/torch_mlp.py` - PyTorch MLP with device support
- `src/classiflow/training/hierarchical_cv.py` - Hierarchical training logic
- `HIERARCHICAL_TRAINING.md` - Full documentation

**Key Features:**
- Two-level hierarchical classification (L1 → L2 per branch)
- Patient/slide-level stratified splits (prevents data leakage)
- Automatic device selection: MPS → CUDA → CPU
- Early stopping with validation-based patience
- Hyperparameter grid search in inner CV
- Supports both single-label and hierarchical modes

### 2026-01-13 (Bug Fixes)
- Fixed sklearn/imblearn version conflict
- Removed VarianceThreshold from pipelines
- Fixed AdaptiveSMOTE pipeline compatibility
- Corrected iris dataset column name
- Added environment setup scripts
- Created comprehensive troubleshooting guide
