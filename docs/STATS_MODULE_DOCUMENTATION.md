# Statistics Module Documentation

## Overview

The `classiflow.stats` module provides a comprehensive, publication-ready statistical analysis and visualization framework for multiclass and pairwise comparisons in molecular subtype research.

**Key Features:**
- ✅ Normality testing (Shapiro-Wilk)
- ✅ Parametric tests (Welch t-test, ANOVA, Tukey HSD)
- ✅ Nonparametric tests (Kruskal-Wallis, Dunn post-hoc with multiple testing correction)
- ✅ Effect size calculations (Cohen's d, Cliff's delta, rank-biserial, log2 fold change)
- ✅ Publication-ready Excel workbooks with multiple sheets
- ✅ Statistical visualizations (boxplots, volcano plots, fold-change plots, heatmaps)
- ✅ Backward compatibility with existing scripts
- ✅ Package-integrated with CLI and Streamlit support

---

## Package Structure

```
src/classiflow/stats/
├── __init__.py           # Public API exports
├── config.py             # StatsConfig and VizConfig dataclasses
├── preprocess.py         # Data preparation and feature selection
├── normality.py          # Shapiro-Wilk normality testing
├── tests.py              # Parametric and nonparametric statistical tests
├── effects.py            # Effect size calculations
├── reports.py            # Publication-ready table builders
├── excel.py              # Excel workbook generation with formatting
├── viz.py                # Statistical visualizations
└── api.py                # Main public API functions

src/classiflow/cli/
└── stats.py              # CLI commands (classiflow stats run/viz)

src/classiflow/streamlit_app/pages/
└── 02_Statistics.py      # Updated Streamlit page

tests/stats/
├── test_preprocess.py    # Tests for data preparation
├── test_normality.py     # Tests for normality testing
└── test_effects.py       # Tests for effect sizes
```

---

## Public API

### 1. Statistical Analysis

```python
from classiflow.stats import run_stats

results = run_stats(
    data_csv="data/mydata.csv",
    label_col="diagnosis",
    outdir="derived/stats_results",
    classes=None,              # Optional: subset/order of classes
    alpha=0.05,                # Significance threshold
    min_n=3,                   # Minimum n per class for Shapiro-Wilk
    dunn_adjust="holm",        # P-value adjustment method
    top_n_features=30,         # Top features in summary sheets
    write_legacy_csv=True,     # Write CSV mirrors
    write_legacy_xlsx=True,    # Write legacy stats_results.xlsx
    fc_center="median",        # Fold-change center (mean or median)
    fc_eps=1e-9,              # Pseudocount for fold-change
)
```

**Returns:**
```python
{
    'publication_xlsx': Path(...),      # Main publication workbook
    'legacy_xlsx': Path(...),           # Legacy workbook
    'stats_dir': Path(...),             # Output directory
    'n_features': int,
    'n_classes': int,
    'n_samples': int,
    'classes': List[str],
    'features': List[str],
    'pairwise_summary': pd.DataFrame,   # Pairwise comparisons
    'top_features_overall': pd.DataFrame # Top ranked features
}
```

### 2. Visualizations

```python
from classiflow.stats import run_visualizations

viz_results = run_visualizations(
    data_csv="data/mydata.csv",
    label_col="diagnosis",
    outdir="derived/viz",
    stats_dir="derived/stats_results",  # Optional: for volcano p-values
    classes=None,
    alpha=0.05,
    fc_thresh=1.0,              # |log2FC| threshold for volcano
    fc_center="median",
    label_topk=12,              # Features to label on volcano
    heatmap_topn=30,            # Top features for heatmap (0=skip)
    fig_dpi=160,
)
```

**Returns:**
```python
{
    'viz_dir': Path(...),
    'boxplots': {'pdf': Path(...), 'png_dir': Path(...)},
    'foldchange': [Path(...)],  # List of CSV + PNG paths
    'volcano': [Path(...)],     # List of CSV + PNG paths
    'heatmap': Path(...) or None
}
```

---

## CLI Usage

### Statistical Analysis

```bash
# Basic analysis
classiflow stats run --data-csv data.csv --label-col diagnosis

# With custom parameters
classiflow stats run \
    --data-csv data.csv \
    --label-col diagnosis \
    --alpha 0.01 \
    --dunn-adjust fdr_bh \
    --top-n 50 \
    --outdir results/
```

### Visualizations

```bash
# Basic visualization
classiflow stats viz --data-csv data.csv --label-col diagnosis

# With stats results
classiflow stats viz \
    --data-csv data.csv \
    --label-col diagnosis \
    --stats-dir results/stats_results \
    --fc-thresh 1.5 \
    --heatmap-topn 50
```

### Available Options

**`classiflow stats run`**
- `--data-csv`: Path to CSV with features + labels (required)
- `--label-col`: Name of label column (required)
- `--outdir`: Output directory (default: `derived`)
- `--classes`: Subset/order of classes (optional, space-separated)
- `--alpha`: Significance threshold (default: 0.05)
- `--min-n`: Minimum n per class for Shapiro-Wilk (default: 3)
- `--dunn-adjust`: P-value adjustment method (default: `holm`)
  - Options: `holm`, `bonferroni`, `fdr_bh`, `fdr_by`, `sidak`
- `--top-n`: Number of top features in summary (default: 30)
- `--no-legacy-csv`: Skip legacy CSV outputs
- `--no-legacy-xlsx`: Skip legacy xlsx output

**`classiflow stats viz`**
- `--data-csv`: Path to CSV (required)
- `--label-col`: Label column name (required)
- `--outdir`: Output directory (default: `derived`)
- `--stats-dir`: Directory with stats results (optional)
- `--classes`: Subset/order of classes (optional)
- `--alpha`: Significance threshold for volcano (default: 0.05)
- `--fc-thresh`: |log2FC| threshold for volcano (default: 1.0)
- `--fc-center`: Center for fold-change (default: `median`)
- `--label-topk`: Top features to annotate on volcano (default: 12)
- `--heatmap-topn`: Top features for heatmap, 0=skip (default: 30)
- `--fig-dpi`: Figure DPI (default: 160)

---

## Output Files

### Publication Workbook (`publication_stats.xlsx`)

**Sheet 1: Run_Manifest**
- Dataset filename, label column, timestamp
- Alpha, min_n, p-adjustment method
- Package version
- Class counts

**Sheet 2: Descriptives_By_Class**
- For each feature × class: n, n_missing, mean, sd, median, q25, q75, IQR

**Sheet 3-4: Normality**
- `Normality_Summary`: Per-feature normality flag + minimum p-value
- `Normality_By_Class`: Shapiro-Wilk results per class

**Sheet 5-6: Parametric Tests**
- `Omnibus_Parametric`: Welch t-test (k=2) or ANOVA (k≥3)
- `PostHoc_Tukey`: Tukey HSD post-hoc (k≥3 only)

**Sheet 7-8: Nonparametric Tests**
- `Omnibus_Nonparametric`: Kruskal-Wallis (always performed)
- `PostHoc_Dunn`: Dunn post-hoc with p-adjustment (always performed)

**Sheet 9: Pairwise_Summary (NEW)**
- One row per (feature, group1, group2)
- Columns: normality, log2fc, fc_center1, fc_center2, cohen_d, cliff_delta, rank_biserial, p_adj, reject

**Sheet 10-11: Top Features (NEW)**
- `Top_Features_Per_Pair`: Top N features per pairwise comparison
- `Top_Features_Overall`: Features ranked by minimum p_adj across all pairs

### Legacy Outputs (Backward Compatibility)

**`stats_results.xlsx`** (legacy format)
- Contains: Normality_Summary, Normality_By_Class, Parametric_Overall, Parametric_PostHoc, Nonparametric_Overall, Nonparametric_PostHoc

**CSV files:**
- `Normality_Summary.csv`
- `Normality_By_Class.csv`
- `Parametric_Overall.csv`
- `Parametric_PostHoc.csv`
- `Nonparametric_Overall.csv`
- `Nonparametric_PostHoc.csv`

### Visualizations

**`derived/viz/boxplots/`**
- `<feature>.png` (one per feature)
- `boxplots_all.pdf` (multipage PDF)

**`derived/viz/foldchange/`**
- `foldchange_<g1>_vs_<g2>.csv` (data table)
- `foldchange_<g1>_vs_<g2>.png` (bar plot)

**`derived/viz/volcano/`**
- `volcano_<g1>_vs_<g2>.csv` (data table)
- `volcano_<g1>_vs_<g2>.png` (volcano plot)
- `volcano_all_pairs.csv` (combined data)

**`derived/viz/heatmaps/`**
- `top_features_heatmap.png` (z-scored heatmap)

---

## Streamlit Integration

The updated Streamlit page [02_Statistics.py](src/classiflow/streamlit_app/pages/02_Statistics.py) provides an interactive UI:

1. **File Upload**: Upload CSV for analysis
2. **Configuration Panel**:
   - Label column selection
   - Significance level (α)
   - Minimum n per class
   - P-value adjustment method
   - Top N features
   - Optional class subset/order
3. **Analysis Buttons**:
   - "Run Statistical Tests" → Executes `run_stats()`
   - "Generate Visualizations" → Executes `run_visualizations()`
4. **Results Display**:
   - Download publication workbook
   - Download legacy workbook
   - Preview top features table
   - Download individual CSV files

---

## Testing

Run tests with pytest:

```bash
# Run all stats tests
pytest tests/stats/

# Run specific test file
pytest tests/stats/test_preprocess.py -v

# Run with coverage
pytest tests/stats/ --cov=classiflow.stats --cov-report=html
```

**Test Coverage:**
- ✅ Feature selection and preprocessing
- ✅ Normality testing (Shapiro-Wilk)
- ✅ Effect size calculations
- ✅ Data validation and error handling
- ✅ Edge cases (insufficient data, NaN handling, constant values)

---

## Backward Compatibility

The new module **preserves backward compatibility**:

1. **Legacy CSV filenames**: Identical to original script outputs
2. **Legacy Excel workbook**: `stats_results.xlsx` with original sheet names
3. **Legacy wrappers**: Can still shell out to scripts if needed
4. **Streamlit page**: Updated but maintains expected outputs

**Migration Path:**
- Existing code using `utils/wrappers.run_stats()` → Update to `classiflow.stats.run_stats()`
- Existing scripts → Replace with CLI: `classiflow stats run ...`
- Streamlit page → Already migrated to new API

---

## Advanced Usage

### Custom Feature Selection

```python
from classiflow.stats import run_stats

results = run_stats(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="results/",
    feature_whitelist=["gene1", "gene2", "gene3"],  # Only analyze these
    # OR
    feature_blacklist=["sample_id", "batch"],       # Exclude these
)
```

### Config-Based Workflow

```python
from classiflow.stats.config import StatsConfig
from classiflow.stats.api import run_stats_from_config

config = StatsConfig(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="results/",
    alpha=0.01,
    dunn_adjust="fdr_bh",
    top_n_features=50,
)

results = run_stats_from_config(config)
```

### Access Intermediate Results

```python
results = run_stats(...)

# Access pairwise summary dataframe
pairwise_df = results['pairwise_summary']
significant = pairwise_df[pairwise_df['reject'] == True]

# Access top features
top_features = results['top_features_overall']
print(top_features.head(10))
```

---

## Dependencies

**Required (automatically installed with `pip install -e ".[stats]"`):**
- `scipy>=1.10.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `statsmodels>=0.14.0`
- `scikit-posthocs>=0.7.0` (required for Dunn test)
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `xlsxwriter>=3.1.0`

---

## Examples

### Example 1: Basic Analysis

```python
from classiflow.stats import run_stats

results = run_stats(
    data_csv="data/gene_expression.csv",
    label_col="tumor_type",
    outdir="results/",
)

print(f"Analyzed {results['n_features']} features across {results['n_classes']} classes")
print(f"Results saved to: {results['publication_xlsx']}")
```

### Example 2: Binary Comparison

```python
from classiflow.stats import run_stats

results = run_stats(
    data_csv="data/treatment_response.csv",
    label_col="response",
    classes=["responder", "non-responder"],  # Binary comparison
    alpha=0.01,
    outdir="results/binary/",
)

# For k=2, you'll get:
# - Welch t-test (parametric)
# - Kruskal-Wallis + Dunn (nonparametric)
# - Pairwise summary with effect sizes
```

### Example 3: Multiclass with Restricted Classes

```python
results = run_stats(
    data_csv="data/subtypes.csv",
    label_col="subtype",
    classes=["TypeA", "TypeB", "TypeC"],  # Only these 3
    dunn_adjust="fdr_bh",                 # Use FDR correction
    top_n_features=50,
    outdir="results/multiclass/",
)
```

### Example 4: Complete Workflow

```python
from classiflow.stats import run_stats, run_visualizations

# Step 1: Run stats
stats_results = run_stats(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="results/",
    alpha=0.05,
    dunn_adjust="holm",
)

# Step 2: Generate visualizations
viz_results = run_visualizations(
    data_csv="data.csv",
    label_col="diagnosis",
    outdir="results/",
    stats_dir=stats_results['stats_dir'],
    fc_thresh=1.5,
    heatmap_topn=50,
)

print("Analysis complete!")
print(f"  Stats: {stats_results['publication_xlsx']}")
print(f"  Viz: {viz_results['viz_dir']}")
```

---

## Troubleshooting

### Error: "scikit-posthocs not installed"

**Solution:**
```bash
pip install scikit-posthocs
# OR
pip install -e ".[stats]"
```

### Error: "Label column not found"

**Solution:** Check that your CSV has the specified label column:
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
```

### Error: "At least 2 classes required"

**Solution:** Ensure your data has at least 2 distinct class labels after filtering.

### Warning: "No numeric features found"

**Solution:** Ensure your CSV has numeric columns besides the label column.

---

## Performance Notes

- **Shapiro-Wilk**: Automatically subsamples to 5000 if n > 5000 per class
- **Dunn post-hoc**: Scales as O(k²) for k classes; efficient for k ≤ 10
- **Visualizations**: PNG generation is parallelizable (considers future optimization)
- **Excel writing**: Uses `xlsxwriter` for efficient formatted output

---

## Citation

If you use this module in your research, please cite:

```
Markowitz, A. (2025). MLSubtype: Production-grade ML toolkit for molecular
subtype classification. https://github.com/alexmarkowitz/classiflow
```

---

## Contributing

To extend the stats module:

1. Add new statistical tests in `tests.py`
2. Add new effect sizes in `effects.py`
3. Update report builders in `reports.py`
4. Add new sheet types in `excel.py`
5. Write tests in `tests/stats/`
6. Update this documentation

---

## License

MIT License - See [LICENSE](LICENSE) file for details.
