# Statistics Module Migration Guide

## Overview

This guide helps you migrate from the old script-based statistics workflow to the new package-integrated `classiflow.stats` module.

---

## Quick Migration Checklist

- [x] Stats subsystem created under `src/classiflow/stats/`
- [x] CLI commands added: `classiflow stats run`, `classiflow stats viz`
- [x] Streamlit page updated to use new API
- [x] Legacy output formats preserved for backward compatibility
- [x] Tests written for core functionality
- [x] Dependencies added to `pyproject.toml`

---

## What Changed

### Before (Old Workflow)

**Scripts:**
- `scripts/stats_normality_and_tests.py` (standalone script)
- `scripts/visualize_stats.py` (standalone script)
- `scripts/umap_plot.py` (standalone script)

**Streamlit:**
- `pages/02_Statistics.py` (shelled out to scripts via `utils/wrappers.py`)

**Usage:**
```python
from utils.wrappers import run_stats, run_visualize_stats

run_stats(root, csv_path, label_col, outdir, extra=flags)
run_visualize_stats(root, csv_path, label_col, outdir, stats_dir)
```

### After (New Workflow)

**Package Module:**
- `src/classiflow/stats/` (importable package with 10 modules)

**CLI:**
```bash
classiflow stats run --data-csv data.csv --label-col diagnosis
classiflow stats viz --data-csv data.csv --label-col diagnosis
```

**Python API:**
```python
from classiflow.stats import run_stats, run_visualizations

results = run_stats(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="derived/stats_results"
)

viz_results = run_visualizations(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="derived/viz",
    stats_dir="derived/stats_results"
)
```

**Streamlit:**
- `src/classiflow/streamlit_app/pages/02_Statistics.py` (uses new API directly)

---

## Migration Steps

### Step 1: Update Dependencies

```bash
# Install stats extras
pip install -e ".[stats]"

# Or install individual packages
pip install statsmodels scikit-posthocs seaborn xlsxwriter
```

### Step 2: Update Imports

**Old:**
```python
from utils.wrappers import run_stats, run_visualize_stats
```

**New:**
```python
from classiflow.stats import run_stats, run_visualizations
```

### Step 3: Update Function Calls

**Old:**
```python
run_stats(
    project_root=root,
    data_csv=csv_path,
    label_col=label_col,
    outdir=outdir,
    classes=classes,
    alpha=alpha,
    min_n=min_n,
    dunn_adjust=dunn_adjust,
    extra=["--flag1", "value1"]  # Extra flags as list
)
```

**New:**
```python
results = run_stats(
    data_csv=csv_path,           # No project_root needed
    label_col=label_col,
    outdir=outdir,
    classes=classes,
    alpha=alpha,
    min_n=min_n,
    dunn_adjust=dunn_adjust,
    # All parameters are explicit, no "extra" flags
    top_n_features=30,
    write_legacy_csv=True,
    write_legacy_xlsx=True,
)

# Results object returned with metadata
print(f"Saved to: {results['publication_xlsx']}")
```

**Old:**
```python
run_visualize_stats(
    project_root=root,
    data_csv=csv_path,
    label_col=label_col,
    outdir=outdir,
    stats_dir=stats_dir,
    classes=classes,
    alpha=alpha,
    fc_thresh=1.0,
    # ... other params
)
```

**New:**
```python
viz_results = run_visualizations(
    data_csv=csv_path,           # No project_root needed
    label_col=label_col,
    outdir=outdir,
    stats_dir=stats_dir,
    classes=classes,
    alpha=alpha,
    fc_thresh=1.0,
    # ... other params (same names)
)

# Results object returned
print(f"Visualizations: {viz_results['viz_dir']}")
```

### Step 4: Update CLI Scripts

**Old:**
```bash
python scripts/stats_normality_and_tests.py \
    --data-csv data.csv \
    --label-col diagnosis \
    --outdir derived \
    --alpha 0.05 \
    --min-n 3 \
    --dunn-adjust holm
```

**New:**
```bash
classiflow stats run \
    --data-csv data.csv \
    --label-col diagnosis \
    --outdir derived \
    --alpha 0.05 \
    --min-n 3 \
    --dunn-adjust holm
```

**Old:**
```bash
python scripts/visualize_stats.py \
    --data-csv data.csv \
    --label-col diagnosis \
    --outdir derived \
    --stats-dir derived/stats_results
```

**New:**
```bash
classiflow stats viz \
    --data-csv data.csv \
    --label-col diagnosis \
    --outdir derived \
    --stats-dir derived/stats_results
```

### Step 5: Update Streamlit Page

The Streamlit page has been **automatically migrated**. The new version:
- Uses `classiflow.stats` API directly (no subprocess calls)
- Provides explicit UI controls for all parameters
- Shows analysis results in session state
- Offers download buttons for publication workbook

**If you have a custom page**, update it like this:

**Old:**
```python
from utils.wrappers import run_stats

if st.button("Run"):
    run_stats(root, csv_path, label_col, DERIVED, extra=extra.split())
```

**New:**
```python
from classiflow.stats import run_stats

if st.button("Run"):
    results = run_stats(
        data_csv=csv_path,
        label_col=label_col,
        outdir=DERIVED,
        alpha=alpha,
        min_n=min_n,
        dunn_adjust=dunn_adjust,
    )
    st.success("Complete!")
    st.write(f"Results: {results['publication_xlsx']}")
```

---

## Backward Compatibility

### What's Preserved

✅ **Output Filenames:**
- `stats_results.xlsx` (legacy workbook)
- `Normality_Summary.csv`
- `Normality_By_Class.csv`
- `Parametric_Overall.csv`
- `Parametric_PostHoc.csv`
- `Nonparametric_Overall.csv`
- `Nonparametric_PostHoc.csv`

✅ **Sheet Names:**
- All legacy sheet names preserved in `stats_results.xlsx`

✅ **Column Names:**
- Identical to original script outputs

✅ **Test Logic:**
- Same statistical tests (Shapiro, Welch, ANOVA, Tukey, Kruskal-Wallis, Dunn)
- Same normality determination logic
- Same p-value adjustment methods

### What's New

✨ **Publication Workbook (`publication_stats.xlsx`):**
- `Run_Manifest` sheet (metadata)
- `Descriptives_By_Class` sheet (summary stats)
- `Pairwise_Summary` sheet (effect sizes + p-values)
- `Top_Features_Per_Pair` sheet
- `Top_Features_Overall` sheet

✨ **Python API Returns Data:**
```python
results = run_stats(...)
# Access results programmatically
pairwise_df = results['pairwise_summary']
top_features = results['top_features_overall']
```

✨ **Improved Excel Formatting:**
- Frozen header rows
- Autofilters enabled
- Numeric formatting (scientific notation for p-values)
- Column autosizing

---

## Breaking Changes

### 1. No `project_root` Parameter

**Old:**
```python
run_stats(project_root=root, data_csv=csv_path, ...)
```

**New:**
```python
run_stats(data_csv=csv_path, ...)  # project_root removed
```

**Reason:** The new API is package-based and doesn't need to locate scripts.

### 2. No `extra` Flags Parameter

**Old:**
```python
run_stats(..., extra=["--custom-flag", "value"])
```

**New:**
```python
# All parameters are explicit:
run_stats(..., top_n_features=30, write_legacy_csv=True, ...)
```

**Reason:** All parameters are now explicit in the function signature for better type safety and IDE support.

### 3. Function Returns Results Dict

**Old:**
```python
run_stats(...)  # Returns None, prints output
```

**New:**
```python
results = run_stats(...)  # Returns dict with metadata
print(results['publication_xlsx'])
print(results['n_features'])
```

**Reason:** Enables programmatic access to results and metadata.

---

## Old Scripts Still Work

If you need to keep using the old scripts temporarily:

```bash
# Old scripts still exist
python scripts/stats_normality_and_tests.py --data-csv data.csv --label-col diagnosis
python scripts/visualize_stats.py --data-csv data.csv --label-col diagnosis
python scripts/umap_plot.py --data-csv data.csv --label-col MOLECULAR
```

However, **we recommend migrating to the new API** for:
- ✅ Better error handling
- ✅ Programmatic access to results
- ✅ Type hints and IDE support
- ✅ Pytest test coverage
- ✅ Package-integrated (no PATH issues)
- ✅ Richer publication workbook

---

## Deprecation Timeline

| Version | Status |
|---------|--------|
| 0.1.0 | Old scripts + new API coexist |
| 0.2.0 | Old scripts marked deprecated |
| 0.3.0 | Old scripts moved to `scripts/legacy/` |
| 1.0.0 | Old scripts removed |

**Recommendation:** Migrate now to avoid breaking changes in future versions.

---

## Common Migration Patterns

### Pattern 1: Batch Analysis Script

**Old:**
```python
import subprocess
from pathlib import Path

datasets = ["data1.csv", "data2.csv", "data3.csv"]

for dataset in datasets:
    subprocess.run([
        "python", "scripts/stats_normality_and_tests.py",
        "--data-csv", dataset,
        "--label-col", "diagnosis",
        "--outdir", f"results/{Path(dataset).stem}/"
    ])
```

**New:**
```python
from classiflow.stats import run_stats
from pathlib import Path

datasets = ["data1.csv", "data2.csv", "data3.csv"]

for dataset in datasets:
    results = run_stats(
        data_csv=dataset,
        label_col="diagnosis",
        outdir=f"results/{Path(dataset).stem}/"
    )
    print(f"✓ {dataset}: {results['n_features']} features, {results['n_classes']} classes")
```

### Pattern 2: Custom Analysis Pipeline

**Old:**
```python
# Run stats
subprocess.run([...stats script...])

# Parse CSV outputs manually
import pandas as pd
dunn_df = pd.read_csv("derived/stats_results/Nonparametric_PostHoc.csv")
significant = dunn_df[dunn_df['p_adj'] < 0.05]
```

**New:**
```python
from classiflow.stats import run_stats

# Run stats and get results directly
results = run_stats(data_csv="data.csv", label_col="diagnosis", outdir="derived/")

# Access pairwise summary with effect sizes
pairwise = results['pairwise_summary']
significant = pairwise[pairwise['reject'] == True]

# Top features already computed
top_features = results['top_features_overall']
print(top_features.head(10))
```

### Pattern 3: Streamlit Integration

**Old:**
```python
import streamlit as st
from utils.wrappers import run_stats

if st.button("Run"):
    with st.spinner("Running..."):
        run_stats(root, csv_path, label_col, outdir)
    st.success("Done")
```

**New:**
```python
import streamlit as st
from classiflow.stats import run_stats

if st.button("Run"):
    with st.spinner("Running..."):
        results = run_stats(
            data_csv=csv_path,
            label_col=label_col,
            outdir=outdir
        )

    st.success(f"✓ Analyzed {results['n_features']} features")

    # Show top features
    st.dataframe(results['top_features_overall'])

    # Download button
    st.download_button(
        "Download Results",
        data=results['publication_xlsx'].read_bytes(),
        file_name="publication_stats.xlsx"
    )
```

---

## FAQ

### Q: Do I need to change my data format?

**A:** No. The new module accepts the same CSV format as before.

### Q: Will my old notebooks still work?

**A:** Old scripts will work until version 1.0.0, but we recommend migrating to avoid deprecation warnings.

### Q: Can I still get the old `stats_results.xlsx` format?

**A:** Yes! Set `write_legacy_xlsx=True` (default) to generate both the new publication workbook and the legacy format.

### Q: What if I need a feature from the old script?

**A:** The new API exposes all parameters explicitly. If something is missing, please open an issue.

### Q: How do I run tests on the new module?

**A:**
```bash
pytest tests/stats/ -v
```

### Q: Can I use both APIs in the same project?

**A:** Yes, during the migration period both APIs will work, but pick one to avoid confusion.

---

## Support

If you encounter issues during migration:

1. Check this guide
2. Review [STATS_MODULE_DOCUMENTATION.md](STATS_MODULE_DOCUMENTATION.md)
3. Run tests: `pytest tests/stats/ -v`
4. Open an issue: https://github.com/alexmarkowitz/classiflow/issues

---

## Summary

The new `classiflow.stats` module is a **drop-in replacement** with:
- ✅ Same statistical tests
- ✅ Same output formats (+ new publication workbook)
- ✅ Better API design
- ✅ Package integration
- ✅ CLI commands
- ✅ Pytest coverage

**Recommended migration:** Update imports, replace `run_stats()` calls, enjoy better API!
