# Compare Multiple Runs

This guide covers comparing models, configurations, or datasets fairly.

## Loading Multiple Runs

```python
import pandas as pd
from pathlib import Path

def load_run_metrics(run_dir):
    """Load summary metrics from a run."""
    run_dir = Path(run_dir)
    summary_file = run_dir / "summary_metrics.csv"

    if summary_file.exists():
        return pd.read_csv(summary_file)
    return None

# Load multiple runs
runs = {
    "Baseline": "derived/run_baseline",
    "SMOTE": "derived/run_smote",
    "Custom Features": "derived/run_custom",
}

all_metrics = {}
for name, path in runs.items():
    metrics = load_run_metrics(path)
    if metrics is not None:
        all_metrics[name] = metrics
```

## Comparison Table

```python
def create_comparison_table(all_metrics, metrics_to_compare=None):
    """Create a comparison table across runs."""

    if metrics_to_compare is None:
        metrics_to_compare = ["roc_auc", "accuracy", "f1", "balanced_accuracy"]

    rows = []
    for run_name, summary in all_metrics.items():
        row = {"Run": run_name}
        for metric in metrics_to_compare:
            metric_row = summary[summary["metric"] == metric]
            if not metric_row.empty:
                mean = metric_row["mean"].values[0]
                std = metric_row["std"].values[0]
                row[metric] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[metric] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows)

comparison = create_comparison_table(all_metrics)
print(comparison.to_markdown(index=False))
```

## Statistical Comparison

### Paired t-test

When comparing on the same folds:

```python
from scipy import stats
import numpy as np

def compare_runs_paired(run1_dir, run2_dir, metric="roc_auc"):
    """Paired t-test comparing two runs on the same folds."""

    def load_fold_values(run_dir, metric):
        values = []
        for fold_dir in sorted(Path(run_dir).glob("fold_*")):
            metrics_file = fold_dir / "metrics_outer_eval.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                val = df[df["metric"] == metric]["value"]
                if not val.empty:
                    values.append(val.values[0])
        return np.array(values)

    values1 = load_fold_values(run1_dir, metric)
    values2 = load_fold_values(run2_dir, metric)

    if len(values1) != len(values2):
        raise ValueError("Runs have different number of folds")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)

    # Effect size (Cohen's d for paired samples)
    diff = values1 - values2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "mean_diff": np.mean(diff),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
    }

result = compare_runs_paired("derived/run_baseline", "derived/run_smote")
print(f"Mean difference: {result['mean_diff']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Cohen's d: {result['cohens_d']:.3f}")
print(f"Significant: {result['significant']}")
```

### Wilcoxon Signed-Rank Test

Non-parametric alternative:

```python
def compare_runs_wilcoxon(run1_dir, run2_dir, metric="roc_auc"):
    """Wilcoxon signed-rank test for paired samples."""

    values1 = load_fold_values(run1_dir, metric)
    values2 = load_fold_values(run2_dir, metric)

    stat, p_value = stats.wilcoxon(values1, values2)

    return {
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
```

## Visualization

### Bar Chart Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_bars(all_metrics, metric="roc_auc", ax=None):
    """Bar chart comparing runs."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    run_names = list(all_metrics.keys())
    means = []
    stds = []

    for name in run_names:
        summary = all_metrics[name]
        row = summary[summary["metric"] == metric]
        if not row.empty:
            means.append(row["mean"].values[0])
            stds.append(row["std"].values[0])
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(run_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue",
                  edgecolor="black", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparison: {metric.upper()}")

    # Add significance markers
    # (You would add asterisks between significant pairs)

    return ax

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric in zip(axes, ["roc_auc", "accuracy", "f1"]):
    plot_comparison_bars(all_metrics, metric=metric, ax=ax)
plt.tight_layout()
plt.savefig("results/comparison_bars.png", dpi=150)
```

### Box Plots Across Folds

```python
def plot_fold_comparison(run_dirs, metric="roc_auc"):
    """Box plots showing fold-level variation."""

    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []

    for name, run_dir in run_dirs.items():
        values = load_fold_values(run_dir, metric)
        data.append(values)
        labels.append(name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Style boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Per-Fold {metric.upper()} Distribution")

    return fig

fig = plot_fold_comparison(runs)
plt.savefig("results/fold_comparison.png", dpi=150)
```

## Fair Comparison Checklist

When comparing runs, ensure:

!!! warning "Use Same Folds"
    Compare runs trained on identical train/test splits (same random seed).

!!! warning "Use Same Data"
    Data preprocessing should be identical or differences documented.

!!! tip "Report All Metrics"
    Don't cherry-pick. Report multiple metrics to show full picture.

!!! tip "Include Variance"
    Always show standard deviation or confidence intervals.

!!! note "Correct for Multiple Comparisons"
    If comparing many runs, adjust p-values (Bonferroni, FDR).

## Comparing Different Datasets

When data differs between runs:

```python
def document_data_differences(run_dirs):
    """Document differences between run datasets."""
    import json

    comparison = []
    for name, run_dir in run_dirs.items():
        manifest_file = Path(run_dir) / "run.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)

            comparison.append({
                "Run": name,
                "Samples": manifest.get("training_data_row_count", "N/A"),
                "Features": len(manifest.get("feature_list", [])),
                "Data Hash": manifest.get("training_data_hash", "N/A")[:16] + "...",
            })

    return pd.DataFrame(comparison)
```

## Reporting Comparisons

Example text for papers:

> We compared baseline models (no SMOTE) with SMOTE-augmented models using 5-fold nested cross-validation. SMOTE significantly improved balanced accuracy (0.831 ± 0.048 vs. 0.759 ± 0.054, paired t-test p=0.023, Cohen's d=0.92). All comparisons used identical fold splits to ensure valid pairing.
