# SMOTE Comparison Quick Start

## TL;DR

```bash
# 1. Train with both SMOTE and no-SMOTE
classiflow train-meta --data-csv data.csv --label-col diagnosis --smote both --outdir derived/results

# 2. Compare performance
classiflow compare-smote derived/results --outdir smote_analysis

# 3. Review recommendation
cat smote_analysis/smote_comparison_*.txt
```

---

## What This Does

The SMOTE comparison feature analyzes your training results to determine whether SMOTE (Synthetic Minority Over-sampling Technique) improves model performance without introducing overfitting.

**Key Outputs**:
- ✓ Statistical comparison (p-values, effect sizes)
- ✓ Overfitting detection (concurrent performance drops)
- ✓ Clear recommendation: USE_SMOTE, NO_SMOTE, or EQUIVALENT
- ✓ Publication-ready plots and reports

---

## Step-by-Step Workflow

### Step 1: Train Models with Both Variants

**Meta-Classifier**:
```bash
classiflow train-meta \
  --data-csv data/MBmerged-z-scores_MLready_correction.csv \
  --label-col MOLECULAR \
  --smote both \
  --outdir derived/results \
  --outer-folds 3 \
  --verbose
```

**Binary Task**:
```bash
classiflow train \
  --data-csv data.csv \
  --label-col diagnosis \
  --smote both \
  --outdir derived/binary_results
```

**Hierarchical** (requires two separate runs):
```bash
# With SMOTE
classiflow train-hierarchical \
  --data-csv data.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --use-smote \
  --outdir derived/hier_smote

# Without SMOTE
classiflow train-hierarchical \
  --data-csv data.csv \
  --patient-col patient_id \
  --label-l1 tumor_type \
  --label-l2 subtype \
  --outdir derived/hier_no_smote
```

**What Happens**:
- Training runs twice per fold: once with SMOTE, once without
- Results are saved with "sampler" column indicating variant
- Adds ~2x training time but provides rigorous comparison

---

### Step 2: Run Comparison Analysis

**Basic**:
```bash
classiflow compare-smote derived/results
```

**Custom Options**:
```bash
classiflow compare-smote derived/results \
  --outdir smote_analysis \
  --primary-metric f1 \
  --secondary-metric roc_auc \
  --overfitting-threshold 0.03 \
  --significance-level 0.05 \
  --min-effect-size 0.2
```

**What Happens**:
- Loads fold-level results from `derived/results/fold1/`, `fold2/`, etc.
- Computes paired t-tests and effect sizes
- Detects overfitting (concurrent drops in multiple metrics)
- Generates recommendation based on statistical evidence
- Creates publication-ready plots and reports

---

### Step 3: Review Results

**Console Output**:
```
======================================================================
  SMOTE VS NO-SMOTE COMPARISON SUMMARY
======================================================================

Model Type: META
Folds: 3

----------------------------------------------------------------------
PERFORMANCE COMPARISON
----------------------------------------------------------------------

f1:
  SMOTE:     0.8234
  No-SMOTE:  0.7891
  Δ (SMOTE - No-SMOTE): +0.0343 **
  p-value:   0.0123
  Cohen's d: 0.543

----------------------------------------------------------------------
RECOMMENDATION
----------------------------------------------------------------------

✓ USE SMOTE
Confidence: HIGH

Reasoning:
  • Δf1 = +0.0343 (p=0.0123, d=0.543)
  • SMOTE significantly improves f1 with meaningful effect size
  • 5/7 other metrics show same direction
======================================================================
```

**Output Files**:
```
smote_analysis/
├── smote_comparison_20240115_143022.txt        ← Text report
├── smote_comparison_20240115_143022.json       ← JSON results
├── smote_comparison_summary_20240115_143022.csv  ← Metric table
├── smote_comparison_delta_bars.png             ← Performance deltas
├── smote_comparison_identity_grid.png          ← Scatter plots
├── smote_comparison_dist_f1.png                ← Distribution comparison
├── smote_comparison_dist_roc_auc.png
├── smote_comparison_trajectory_f1.png          ← Per-fold trajectories
└── smote_comparison_per_task_f1.png            ← Per-task comparison
```

---

## Interpreting Results

### Recommendation Types

| Recommendation | Meaning | Action |
|----------------|---------|--------|
| **USE_SMOTE** | SMOTE significantly improves performance | Use SMOTE in final model |
| **NO_SMOTE** | No-SMOTE performs better or overfitting detected | Do not use SMOTE |
| **EQUIVALENT** | No significant difference | Either is fine; use simpler (no-SMOTE) |
| **INSUFFICIENT_DATA** | Cannot compute statistics | Need more folds or check data |

### Confidence Levels

| Confidence | Criteria |
|------------|----------|
| **HIGH** | p < 0.05 AND \|Cohen's d\| ≥ 0.2 |
| **MEDIUM** | Only one criterion met |
| **LOW** | Neither criterion met |

### Overfitting Warning

If you see:
```
⚠️  OVERFITTING DETECTED in: f1, roc_auc
Reason: Concurrent drops in f1 and roc_auc suggest SMOTE may be overfitting
```

**Action**: Use **NO_SMOTE** regardless of other metrics. Overfitting means SMOTE is generating synthetic samples that don't generalize.

---

## Common Scenarios

### Scenario 1: Clear Winner

**Output**:
```
Δf1 = +0.045 (p=0.003, d=0.78)
SMOTE significantly improves f1 with meaningful effect size
```

**Action**: Use SMOTE with high confidence

**Publication Statement**:
> "SMOTE significantly improved F1 score (0.85 vs 0.80, p=0.003, Cohen's d=0.78)
> without evidence of overfitting."

---

### Scenario 2: Overfitting Detected

**Output**:
```
Δf1 = -0.04, Δroc_auc = -0.03
⚠️  OVERFITTING DETECTED
```

**Action**: Do NOT use SMOTE

**Publication Statement**:
> "While SMOTE slightly improved training metrics, validation performance
> declined (F1: -0.04, ROC AUC: -0.03), suggesting overfitting to synthetic
> samples. We proceeded without SMOTE."

---

### Scenario 3: Equivalent Performance

**Output**:
```
Δf1 = +0.007 (p=0.21, d=0.12)
No significant or meaningful difference detected
```

**Action**: Use no-SMOTE for simplicity

**Publication Statement**:
> "SMOTE and no-SMOTE variants showed equivalent performance (F1: 0.82 vs 0.81,
> p=0.21). We report no-SMOTE results for simplicity and reproducibility."

---

### Scenario 4: Marginal Improvement

**Output**:
```
Δf1 = +0.02 (p=0.04, d=0.35)
SMOTE significantly improves f1 with meaningful effect size
Confidence: MEDIUM
```

**Action**: Consider using SMOTE, but validate on independent test set

**Publication Statement**:
> "SMOTE provided a modest but significant improvement in F1 score (0.82 vs 0.80,
> p=0.04, Cohen's d=0.35). We validated this improvement on an independent test
> set (F1: 0.81 with SMOTE vs 0.79 without)."

---

## Advanced Options

### Custom Thresholds

**Strict Criteria** (for high-stakes applications):
```bash
classiflow compare-smote derived/results \
  --significance-level 0.01 \
  --min-effect-size 0.5 \
  --overfitting-threshold 0.05
```

**Lenient Criteria** (for exploratory analysis):
```bash
classiflow compare-smote derived/results \
  --significance-level 0.10 \
  --min-effect-size 0.1 \
  --overfitting-threshold 0.01
```

### Alternative Primary Metric

```bash
# Use accuracy instead of F1
classiflow compare-smote derived/results --primary-metric accuracy

# Use ROC AUC
classiflow compare-smote derived/results --primary-metric roc_auc
```

### Skip Plots (Fast Analysis)

```bash
classiflow compare-smote derived/results --no-plots
```

---

## Python API

### Quick Analysis

```python
from classiflow.evaluation import SMOTEComparison

# Load and analyze
comparison = SMOTEComparison.from_directory("derived/results")
result = comparison.generate_report()

# Print summary
print(result.summary_text())

# Save outputs
comparison.save_report(result, "smote_analysis")
comparison.create_all_plots("smote_analysis")
```

### Extract Statistics

```python
comparison = SMOTEComparison.from_directory("derived/results")

# Get per-metric statistics
stats = comparison.compute_statistics()

for metric, values in stats.items():
    print(f"{metric}:")
    print(f"  SMOTE:     {values['smote_mean']:.4f}")
    print(f"  No-SMOTE:  {values['no_smote_mean']:.4f}")
    print(f"  Δ:         {values['delta']:+.4f}")
    print(f"  p-value:   {values['paired_t_pval']:.4f}")
    print(f"  Cohen's d: {values['cohens_d']:.3f}")
```

---

## Troubleshooting

### Error: "No metrics files found"

**Cause**: Wrong directory or missing fold subdirectories

**Fix**:
```bash
# Check structure
ls derived/results/
# Should show: fold1/ fold2/ fold3/

# If using different structure, specify metric file
classiflow compare-smote derived/results --metric-file metrics_inner_cv.csv
```

### Error: "Could not identify SMOTE variants"

**Cause**: Models were not trained with `--smote both`

**Fix**: Retrain with both variants:
```bash
classiflow train-meta --smote both ...
```

### Warning: "Only 1 fold available"

**Cause**: Not enough folds for paired statistical tests

**Fix**: Use ≥3 folds:
```bash
classiflow train-meta --outer-folds 3 --smote both ...
```

---

## Best Practices

### 1. Always Train with `--smote both`

This ensures fair comparison with identical train/test splits:

✅ **Correct**:
```bash
classiflow train-meta --smote both
```

❌ **Incorrect** (separate runs may have different splits):
```bash
classiflow train-meta --smote on
classiflow train-meta --smote off  # Different random splits!
```

### 2. Use ≥3 Folds

Statistical tests require multiple paired samples:
```bash
classiflow train-meta --outer-folds 3 --smote both  # Minimum
classiflow train-meta --outer-folds 5 --smote both  # Better
```

### 3. Check Multiple Metrics

Don't rely on F1 alone:
```bash
classiflow compare-smote derived/results \
  --primary-metric f1 \
  --secondary-metric roc_auc
```

Review all metrics in the report (accuracy, balanced accuracy, MCC, etc.)

### 4. Validate on Independent Test Set

Cross-validation shows internal consistency. Always confirm on held-out data:

```bash
# After comparing, train final model
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

## When to Use SMOTE

| Scenario | Recommendation |
|----------|----------------|
| Balanced classes (< 3:1 ratio) | SMOTE unlikely to help |
| Moderate imbalance (3:1 - 10:1) | Test with `--smote both` |
| Severe imbalance (> 10:1) | SMOTE likely beneficial |
| Overfitting detected | Do NOT use SMOTE |
| Equivalent performance | Prefer no-SMOTE (simpler) |

---

## Next Steps

1. **Run comparison**: `classiflow compare-smote derived/results`
2. **Review plots**: Check `smote_analysis/*.png`
3. **Read full guide**: [SMOTE_COMPARISON_GUIDE.md](SMOTE_COMPARISON_GUIDE.md)
4. **Try examples**: `python example_smote_comparison.py`

---

## Support

- **Full Documentation**: [SMOTE_COMPARISON_GUIDE.md](SMOTE_COMPARISON_GUIDE.md)
- **GitHub Issues**: https://github.com/uclahs-cds/classiflow/issues
- **Email**: boutroslab-support@mednet.ucla.edu
