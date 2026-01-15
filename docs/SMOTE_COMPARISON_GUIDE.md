# SMOTE vs No-SMOTE Comparison Guide

## Overview

The SMOTE comparison feature provides comprehensive tools for evaluating whether SMOTE (Synthetic Minority Over-sampling Technique) improves model performance without introducing overfitting. This is critical for publication-quality machine learning research, where reviewers need evidence-based justification for preprocessing choices.

## Quick Start

### 1. Train with Both Variants

First, train your model with both SMOTE and no-SMOTE variants:

```bash
# Meta-classifier
classiflow train-meta \
  --data-csv data.csv \
  --label-col diagnosis \
  --smote both \
  --outdir derived/results

# Binary task
classiflow train \
  --data-csv data.csv \
  --label-col diagnosis \
  --smote both \
  --outdir derived/binary_results

# Hierarchical (requires running both)
classiflow train-hierarchical \
  --data-csv data.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --use-smote \
  --outdir derived/hier_smote

classiflow train-hierarchical \
  --data-csv data.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --outdir derived/hier_no_smote
```

### 2. Run Comparison

```bash
classiflow compare-smote derived/results --outdir smote_analysis
```

### 3. Review Results

The analysis generates:
- **Text report**: Human-readable summary with recommendation
- **JSON report**: Machine-readable results for further processing
- **CSV summary**: Detailed metric comparisons
- **Publication-ready plots**: Delta charts, scatter plots, distributions, trajectories

---

## What Gets Analyzed

### Statistical Tests

1. **Paired t-test**: Tests if mean performance difference is significant
   - Assumes normal distribution of fold-level differences
   - Appropriate for ≥3 folds

2. **Wilcoxon signed-rank test**: Non-parametric alternative
   - Doesn't assume normality
   - More robust for small sample sizes

3. **Cohen's d (Effect Size)**: Measures practical significance
   - Small: d = 0.2
   - Medium: d = 0.5
   - Large: d = 0.8

### Overfitting Detection

Overfitting is suspected when **both** conditions are met:
- Primary metric drops by ≥ threshold (default: 0.03)
- Secondary metric drops by ≥ threshold

This indicates SMOTE may be generating unrealistic synthetic samples that don't generalize.

### Recommendation Logic

The analysis provides one of four recommendations:

| Recommendation | Conditions |
|----------------|------------|
| **USE_SMOTE** | SMOTE significantly better (p < 0.05) AND meaningful effect (d ≥ 0.2) |
| **NO_SMOTE** | No-SMOTE significantly better OR overfitting detected |
| **EQUIVALENT** | No significant difference AND no meaningful effect |
| **INSUFFICIENT_DATA** | Unable to compute statistics (missing metrics) |

**Confidence Levels**:
- **High**: Both statistical significance (p < 0.05) AND meaningful effect (d ≥ 0.2)
- **Medium**: Only one criterion met
- **Low**: Neither criterion met but trend observed

---

## CLI Reference

### Basic Usage

```bash
classiflow compare-smote <result_dir> [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir` | `smote_analysis` | Output directory for comparison results |
| `--model-type` | Auto-detect | Model type: `binary`, `meta`, or `hierarchical` |
| `--metric-file` | `metrics_outer_meta_eval.csv` | Name of metrics CSV to load |
| `--primary-metric` | `f1` | Primary metric for recommendation |
| `--secondary-metric` | `roc_auc` | Secondary metric for overfitting detection |
| `--overfitting-threshold` | `0.03` | Minimum drop to flag overfitting |
| `--significance-level` | `0.05` | p-value threshold |
| `--min-effect-size` | `0.2` | Minimum Cohen's d for meaningful effect |
| `--no-plots` | `False` | Skip plot generation |
| `--verbose` | `False` | Verbose output |

### Examples

#### Basic Comparison (Meta-Classifier)

```bash
classiflow compare-smote derived/results
```

#### Specify Model Type and Primary Metric

```bash
classiflow compare-smote derived/results \
  --model-type meta \
  --primary-metric f1 \
  --secondary-metric roc_auc
```

#### Custom Thresholds

```bash
classiflow compare-smote derived/results \
  --overfitting-threshold 0.05 \
  --min-effect-size 0.3 \
  --significance-level 0.01
```

#### Hierarchical Models

For hierarchical models, you need to manually combine results:

```bash
# Option 1: Manually combine fold results
mkdir -p derived/hier_combined/fold{1,2,3}
cp derived/hier_smote/fold1/metrics.csv derived/hier_combined/fold1/
cp derived/hier_no_smote/fold1/metrics.csv derived/hier_combined/fold1/
# Repeat for other folds...

classiflow compare-smote derived/hier_combined \
  --model-type hierarchical \
  --metric-file metrics.csv
```

#### Skip Plots (Fast Analysis)

```bash
classiflow compare-smote derived/results --no-plots
```

---

## Output Files

### Reports

| File | Content |
|------|---------|
| `smote_comparison_YYYYMMDD_HHMMSS.txt` | Human-readable report with recommendation |
| `smote_comparison_YYYYMMDD_HHMMSS.json` | Machine-readable results (all statistics) |
| `smote_comparison_summary_YYYYMMDD_HHMMSS.csv` | Metric-by-metric comparison table |

### Plots

| Plot | Description |
|------|-------------|
| `smote_comparison_delta_bars.png` | Horizontal bar chart of performance differences |
| `smote_comparison_identity_grid.png` | Grid of scatter plots for all metrics |
| `smote_comparison_dist_<metric>.png` | Violin plots showing distributions |
| `smote_comparison_trajectory_<metric>.png` | Per-fold performance lines |
| `smote_comparison_per_task_<metric>.png` | Grouped bar chart per task (binary/meta only) |

---

## Python API

### Basic Usage

```python
from classiflow.evaluation.smote_comparison import SMOTEComparison

# Load results
comparison = SMOTEComparison.from_directory("derived/results")

# Generate report
result = comparison.generate_report(
    primary_metric="f1",
    secondary_metric="roc_auc",
    overfitting_threshold=0.03,
)

# Print summary
print(result.summary_text())

# Save reports
comparison.save_report(result, "smote_analysis")

# Create plots
comparison.create_all_plots("smote_analysis")
```

### Advanced: Custom Analysis

```python
from classiflow.evaluation.smote_comparison import SMOTEComparison
import pandas as pd

# Load and filter data
comparison = SMOTEComparison.from_directory("derived/results")

# Compute statistics for specific metrics
stats = comparison.compute_statistics(metrics=["f1", "accuracy", "roc_auc"])

# Access per-metric results
for metric, values in stats.items():
    print(f"{metric}:")
    print(f"  SMOTE mean: {values['smote_mean']:.4f}")
    print(f"  No-SMOTE mean: {values['no_smote_mean']:.4f}")
    print(f"  Delta: {values['delta']:+.4f}")
    print(f"  p-value: {values['paired_t_pval']:.4f}")
    print(f"  Cohen's d: {values['cohens_d']:.3f}")

# Detect overfitting
overfitting, metrics, reason = comparison.detect_overfitting(
    primary_metric="f1",
    secondary_metric="roc_auc",
    delta_threshold=0.03,
)

if overfitting:
    print(f"⚠️  Overfitting detected: {reason}")

# Generate recommendation
stats_dict = comparison.compute_statistics()
recommendation, confidence, reasoning = comparison.generate_recommendation(
    stats_dict,
    primary_metric="f1",
    significance_level=0.05,
    min_effect_size=0.2,
)

print(f"Recommendation: {recommendation} (confidence: {confidence})")
for r in reasoning:
    print(f"  • {r}")
```

### Loading from Custom Data

```python
import pandas as pd
from classiflow.evaluation.smote_comparison import SMOTEComparison

# Load from custom CSVs
smote_data = pd.read_csv("results_smote.csv")
no_smote_data = pd.read_csv("results_no_smote.csv")

# Must have 'fold' column
assert "fold" in smote_data.columns
assert "fold" in no_smote_data.columns

# Create comparison
comparison = SMOTEComparison(
    smote_data=smote_data,
    no_smote_data=no_smote_data,
    model_type="meta",
)

# Run analysis
result = comparison.generate_report()
```

---

## Interpreting Results

### Example Output

```
======================================================================
  SMOTE VS NO-SMOTE COMPARISON SUMMARY
======================================================================

Model Type: META
Folds: 3
Tasks: 14

----------------------------------------------------------------------
PERFORMANCE COMPARISON
----------------------------------------------------------------------

f1:
  SMOTE:     0.8234
  No-SMOTE:  0.7891
  Δ (SMOTE - No-SMOTE): +0.0343 **
  p-value:   0.0123
  Cohen's d: 0.543

roc_auc:
  SMOTE:     0.8912
  No-SMOTE:  0.8756
  Δ (SMOTE - No-SMOTE): +0.0156
  p-value:   0.0891
  Cohen's d: 0.234

----------------------------------------------------------------------
OVERFITTING ANALYSIS
----------------------------------------------------------------------

✓ No overfitting detected

----------------------------------------------------------------------
RECOMMENDATION
----------------------------------------------------------------------

✓ USE SMOTE
Confidence: HIGH

Reasoning:
  • Δf1 = +0.0343 (p=0.0123, d=0.543)
  • SMOTE significantly improves f1 with meaningful effect size
  • 5/7 other metrics show same direction
```

### What to Report in Publications

#### If Recommending SMOTE:

> "We evaluated SMOTE's impact on model performance using 3-fold cross-validation.
> SMOTE significantly improved F1 score (0.823 vs 0.789, p=0.012, Cohen's d=0.54)
> without evidence of overfitting, as secondary metrics (ROC AUC) remained stable
> or improved. All analyses were performed using classiflow v1.0 compare-smote tool."

#### If Recommending No-SMOTE:

> "While SMOTE slightly improved training performance, we observed concurrent
> drops in F1 (-0.034) and ROC AUC (-0.028) during validation, suggesting
> overfitting to synthetic samples (p<0.05). We therefore proceeded with no-SMOTE
> to ensure model generalization."

#### If Equivalent:

> "SMOTE and no-SMOTE variants showed statistically equivalent performance
> (F1: 0.82 vs 0.81, p=0.23, d=0.15). We report no-SMOTE results for simplicity
> and reproducibility."

---

## Best Practices

### 1. Always Use Nested Cross-Validation

SMOTE must be applied **inside** the CV loop to avoid data leakage:

✅ **Correct**: SMOTE applied separately per fold
```bash
classiflow train-meta --smote both  # Applies SMOTE per-fold
```

❌ **Incorrect**: SMOTE before CV split
```python
# Don't do this!
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
# Then split into folds...
```

### 2. Compare Multiple Metrics

Don't rely on a single metric. Check:
- **F1 Score**: Balance of precision/recall
- **ROC AUC**: Ranking quality across thresholds
- **Balanced Accuracy**: Per-class accuracy
- **MCC**: Overall classification quality

### 3. Check for Overfitting

Look for **concurrent drops** in multiple metrics. A single metric drop may be noise.

### 4. Consider Class Imbalance Severity

| Imbalance Ratio | Recommendation |
|-----------------|----------------|
| < 3:1 | SMOTE may not help significantly |
| 3:1 - 10:1 | SMOTE likely beneficial |
| > 10:1 | SMOTE highly recommended, but validate carefully |

### 5. Validate on Independent Test Set

Cross-validation shows internal consistency. Always validate on held-out test data:

```bash
# After comparing SMOTE variants, retrain on full data
classiflow train-meta \
  --data-csv train.csv \
  --label-col diagnosis \
  --smote on \  # Use recommendation
  --outdir final_model

# Test on independent data
classiflow infer \
  --data-csv test.csv \
  --run-dir final_model/fold1 \
  --label-col diagnosis \
  --outdir test_results
```

---

## Troubleshooting

### Error: "No metrics files found"

**Cause**: Result directory doesn't contain fold subdirectories or metrics CSVs

**Solution**:
```bash
# Check directory structure
ls derived/results/
# Should show: fold1/ fold2/ fold3/

ls derived/results/fold1/
# Should show: metrics_outer_meta_eval.csv (or similar)

# If metrics file has different name, specify it
classiflow compare-smote derived/results --metric-file metrics_inner_cv.csv
```

### Error: "Could not identify SMOTE variants"

**Cause**: Metrics CSV doesn't have 'sampler' column with 'smote' and 'none' values

**Solution**: Ensure training was run with `--smote both`:
```bash
classiflow train-meta --smote both ...
```

### Warning: "Only 1 fold available"

**Cause**: Statistical tests require ≥2 folds for paired comparisons

**Solution**: Retrain with more folds:
```bash
classiflow train-meta --outer-folds 3 --smote both ...
```

### Plots Not Generated

**Cause**: Missing optional dependencies (matplotlib, seaborn)

**Solution**:
```bash
pip install matplotlib seaborn
```

---

## FAQ

### Q: Do I need to run training twice for hierarchical models?

**A**: Currently yes. The `--smote` flag in hierarchical mode is boolean (on/off), not "both":

```bash
# SMOTE variant
classiflow train-hierarchical --use-smote ...

# No-SMOTE variant
classiflow train-hierarchical ...  # (no --use-smote flag)
```

### Q: Can I compare more than two variants (e.g., different k_neighbors)?

**A**: Not with the current CLI. Use the Python API to load and compare custom datasets.

### Q: What if my primary metric isn't available?

**A**: Specify an available metric:
```bash
classiflow compare-smote derived/results --primary-metric accuracy
```

### Q: How do I compare specific tasks only?

**A**: Filter data in Python:

```python
from classiflow.evaluation.smote_comparison import SMOTEComparison

comparison = SMOTEComparison.from_directory("derived/results")

# Filter to specific tasks
tasks_to_compare = ["G3_vs_G4", "SHH_vs_WNT"]
smote_filtered = comparison.smote_data[comparison.smote_data["task"].isin(tasks_to_compare)]
no_smote_filtered = comparison.no_smote_data[comparison.no_smote_data["task"].isin(tasks_to_compare)]

# Create new comparison
filtered_comparison = SMOTEComparison(smote_filtered, no_smote_filtered, "meta")
result = filtered_comparison.generate_report()
```

### Q: Can I export results to Excel?

**A**: Yes, using pandas:

```python
result = comparison.generate_report()
result_dict = result.to_dict()

import pandas as pd
df = pd.DataFrame([result_dict])
df.to_excel("smote_comparison.xlsx", index=False)
```

---

## Related Documentation

- [Data Compatibility Guide](DATA_COMPATIBILITY_GUIDE.md) - Ensure data quality before training
- [Custom Tasks Guide](CUSTOM_TASKS_GUIDE.md) - Define custom binary classification tasks
- [Training Guide](README.md#training) - Full training workflow
- [SMOTE Implementation](src/classiflow/models/smote.py) - Technical details

---

## Citation

If you use this SMOTE comparison feature in your research, please cite:

```bibtex
@software{classiflow,
  title = {classiflow: Production-Grade ML Toolkit for Molecular Subtype Classification},
  author = {Boutros Lab},
  year = {2024},
  url = {https://github.com/uclahs-cds/classiflow}
}
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/uclahs-cds/classiflow/issues
- Email: boutroslab-support@mednet.ucla.edu
