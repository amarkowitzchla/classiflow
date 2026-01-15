# Methods Text & Tables Tutorial

This tutorial shows how to generate methods paragraphs and summary tables suitable for peer-reviewed publications.

## What You'll Learn

- Extracting key information from run manifests
- Generating methods text templates
- Creating publication-ready tables
- Exporting to LaTeX and Word-compatible formats

## Step 1: Load Run Information

```python
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("outputs/06_methods_tutorial")
output_dir.mkdir(parents=True, exist_ok=True)

# Load from a completed run
run_dir = Path("outputs/01_binary_tutorial/run")

# Load manifest
with open(run_dir / "run.json") as f:
    manifest = json.load(f)

# Load config
with open(run_dir / "config.json") as f:
    config = json.load(f)

# Load summary metrics
summary = pd.read_csv(run_dir / "summary_metrics.csv")

print("Run information loaded:")
print(f"  Run ID: {manifest['run_id'][:8]}...")
print(f"  Timestamp: {manifest['timestamp']}")
print(f"  Features: {len(manifest.get('feature_list', []))} features")
```

## Step 2: Generate Methods Paragraph

### Automatic Template

```python
def generate_methods_text(manifest, config, summary):
    """Generate a methods paragraph from run information."""

    # Extract key values
    n_features = len(manifest.get("feature_list", []))
    outer_folds = config.get("outer_folds", "N/A")
    inner_splits = config.get("inner_splits", "N/A")
    inner_repeats = config.get("inner_repeats", 1)
    random_state = config.get("random_state", "N/A")
    smote_mode = config.get("smote_mode", "off")

    # Get primary metrics
    auc_row = summary[summary["metric"] == "roc_auc"]
    acc_row = summary[summary["metric"] == "accuracy"]

    auc_mean = auc_row["mean"].values[0] if not auc_row.empty else "N/A"
    auc_std = auc_row["std"].values[0] if not auc_row.empty else "N/A"
    acc_mean = acc_row["mean"].values[0] if not acc_row.empty else "N/A"
    acc_std = acc_row["std"].values[0] if not acc_row.empty else "N/A"

    # SMOTE text
    smote_text = ""
    if smote_mode == "on":
        smote_text = " SMOTE was applied to training folds to address class imbalance."
    elif smote_mode == "both":
        smote_text = " Models were trained with and without SMOTE to assess the impact of class imbalance handling."

    # Package version
    version = manifest.get("package_version", "0.1.0")

    methods = f"""
Machine Learning Methods

Binary classification was performed using classiflow v{version} with {outer_folds}-fold
nested cross-validation. The inner loop used {inner_splits}-fold cross-validation
{"repeated " + str(inner_repeats) + " times " if inner_repeats > 1 else ""}for hyperparameter optimization.
Model selection was performed using area under the ROC curve (AUC) as the primary metric.
Candidate models included logistic regression, support vector machines, random forests,
and gradient boosting classifiers. Feature standardization was applied within each training fold.
{smote_text}

The final model achieved a mean AUC of {auc_mean:.3f} (SD = {auc_std:.3f}) and accuracy
of {acc_mean:.3f} (SD = {acc_std:.3f}) across outer folds. All analyses used a fixed
random seed ({random_state}) for reproducibility. The complete run manifest,
including software versions and data checksums, is provided in the supplementary materials.
"""
    return methods.strip()


methods_text = generate_methods_text(manifest, config, summary)
print(methods_text)

# Save to file
with open(output_dir / "methods_paragraph.txt", "w") as f:
    f.write(methods_text)
```

### Customizable Template

```python
METHODS_TEMPLATE = """
## Machine Learning

### Model Training

Classification was performed using classiflow v{version} [{cite_classiflow}].
We used {outer_folds}-fold nested cross-validation to obtain unbiased performance
estimates while tuning hyperparameters in the inner loop ({inner_splits}-fold CV,
{inner_repeats} repeats).

### Candidate Models

The following model families were evaluated:
- Logistic regression (L2 regularization, C ∈ {{0.01, 0.1, 1, 10, 100}})
- Support vector machines (RBF kernel, C ∈ {{0.1, 1, 10}}, γ ∈ {{'scale', 'auto'}})
- Random forest (n_estimators ∈ {{100, 200}}, max_depth ∈ {{5, 10, None}})
- Gradient boosting (learning_rate ∈ {{0.01, 0.1}}, n_estimators ∈ {{100, 200}})

### Class Imbalance

{smote_description}

### Performance Metrics

Model performance was evaluated using:
- Area under the ROC curve (AUC) - primary metric for model selection
- Accuracy, precision, recall, and F1 score
- Balanced accuracy (for imbalanced datasets)

### Reproducibility

All analyses used random seed {random_state}. Software versions: Python {python_version},
scikit-learn {sklearn_version}. Data integrity was verified using SHA-256 checksums.
"""

def fill_methods_template(manifest, config, summary):
    """Fill the methods template with actual values."""
    import sklearn

    smote_mode = config.get("smote_mode", "off")
    if smote_mode == "off":
        smote_desc = "No resampling was applied."
    elif smote_mode == "on":
        k = config.get("smote_k_neighbors", 5)
        smote_desc = f"SMOTE (k={k}) was applied within training folds to address class imbalance."
    else:
        smote_desc = "Models were trained with and without SMOTE for comparison."

    return METHODS_TEMPLATE.format(
        version=manifest.get("package_version", "0.1.0"),
        cite_classiflow="Markowitz, 2024",
        outer_folds=config.get("outer_folds", 5),
        inner_splits=config.get("inner_splits", 5),
        inner_repeats=config.get("inner_repeats", 1),
        smote_description=smote_desc,
        random_state=config.get("random_state", 42),
        python_version=manifest.get("python_version", "3.10"),
        sklearn_version=sklearn.__version__,
    )

detailed_methods = fill_methods_template(manifest, config, summary)
with open(output_dir / "methods_detailed.md", "w") as f:
    f.write(detailed_methods)
print("Detailed methods saved to methods_detailed.md")
```

## Step 3: Create Summary Tables

### Performance Metrics Table

```python
def create_metrics_table(summary, format="markdown"):
    """Create a formatted metrics table."""

    # Select key metrics
    key_metrics = ["roc_auc", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]
    display_names = {
        "roc_auc": "AUC",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall (Sensitivity)",
    }

    rows = []
    for metric in key_metrics:
        row = summary[summary["metric"] == metric]
        if not row.empty:
            mean = row["mean"].values[0]
            std = row["std"].values[0]
            min_val = row["min"].values[0]
            max_val = row["max"].values[0]

            rows.append({
                "Metric": display_names.get(metric, metric),
                "Mean": f"{mean:.3f}",
                "SD": f"{std:.3f}",
                "Range": f"[{min_val:.3f}, {max_val:.3f}]",
                "Mean ± SD": f"{mean:.3f} ± {std:.3f}",
            })

    df = pd.DataFrame(rows)

    if format == "latex":
        return df.to_latex(index=False, escape=False)
    elif format == "markdown":
        return df.to_markdown(index=False)
    else:
        return df

# Markdown table
md_table = create_metrics_table(summary, format="markdown")
print("Performance Metrics Table (Markdown):\n")
print(md_table)

# Save
with open(output_dir / "table_metrics.md", "w") as f:
    f.write("# Classification Performance Metrics\n\n")
    f.write(md_table)
```

### LaTeX Table

```python
# LaTeX table for direct inclusion in papers
latex_table = create_metrics_table(summary, format="latex")
print("\nPerformance Metrics Table (LaTeX):\n")
print(latex_table)

with open(output_dir / "table_metrics.tex", "w") as f:
    f.write("% Classification Performance Metrics\n")
    f.write("% Generated by classiflow\n\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Classification performance metrics (mean ± SD across 5-fold nested CV)}\n")
    f.write("\\label{tab:metrics}\n")
    f.write(latex_table)
    f.write("\\end{table}\n")
```

## Step 4: Dataset Characteristics Table

```python
def create_dataset_table(manifest, config, data_path=None):
    """Create a table describing the dataset."""

    rows = []

    # From manifest
    if "training_data_row_count" in manifest:
        rows.append(("Total samples", str(manifest["training_data_row_count"])))

    if "feature_list" in manifest:
        rows.append(("Number of features", str(len(manifest["feature_list"]))))

    # From config
    rows.append(("Label column", config.get("label_col", "N/A")))

    if "pos_label" in config:
        rows.append(("Positive class", config["pos_label"]))

    # CV structure
    rows.append(("Outer CV folds", str(config.get("outer_folds", "N/A"))))
    rows.append(("Inner CV folds", str(config.get("inner_splits", "N/A"))))

    if config.get("inner_repeats", 1) > 1:
        rows.append(("Inner CV repeats", str(config["inner_repeats"])))

    rows.append(("Random seed", str(config.get("random_state", "N/A"))))
    rows.append(("SMOTE", config.get("smote_mode", "off")))

    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    return df

dataset_table = create_dataset_table(manifest, config)
print("\nDataset Characteristics:\n")
print(dataset_table.to_markdown(index=False))

with open(output_dir / "table_dataset.md", "w") as f:
    f.write("# Dataset and Experimental Setup\n\n")
    f.write(dataset_table.to_markdown(index=False))
```

## Step 5: Per-Fold Results Table

```python
def create_fold_table(run_dir):
    """Create a table of per-fold results."""

    rows = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        metrics_file = fold_dir / "metrics_outer_eval.csv"
        if metrics_file.exists():
            fold_df = pd.read_csv(metrics_file)
            fold_num = int(fold_dir.name.split("_")[1])

            # Extract key metrics
            row = {"Fold": fold_num}
            for metric in ["roc_auc", "accuracy", "f1"]:
                metric_row = fold_df[fold_df["metric"] == metric]
                if not metric_row.empty:
                    row[metric.upper() if metric == "roc_auc" else metric.capitalize()] = \
                        f"{metric_row['value'].values[0]:.3f}"
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        return df
    return None

fold_table = create_fold_table(run_dir)
if fold_table is not None:
    print("\nPer-Fold Results:\n")
    print(fold_table.to_markdown(index=False))

    with open(output_dir / "table_folds.md", "w") as f:
        f.write("# Per-Fold Classification Results\n\n")
        f.write(fold_table.to_markdown(index=False))
```

## Step 6: Export to Excel

```python
def export_all_tables(output_dir, manifest, config, summary, run_dir):
    """Export all tables to a single Excel workbook."""

    with pd.ExcelWriter(output_dir / "supplementary_tables.xlsx", engine="openpyxl") as writer:
        # Dataset info
        dataset_df = create_dataset_table(manifest, config)
        dataset_df.to_excel(writer, sheet_name="Dataset", index=False)

        # Performance metrics
        metrics_df = pd.DataFrame(create_metrics_table(summary, format="dataframe"))
        summary.to_excel(writer, sheet_name="Metrics Summary", index=False)

        # Per-fold results
        fold_df = create_fold_table(run_dir)
        if fold_df is not None:
            fold_df.to_excel(writer, sheet_name="Per-Fold Results", index=False)

        # Feature list
        if "feature_list" in manifest:
            features_df = pd.DataFrame({
                "Feature": manifest["feature_list"],
                "Index": range(len(manifest["feature_list"]))
            })
            features_df.to_excel(writer, sheet_name="Features", index=False)

    print(f"\nExcel workbook saved to: {output_dir / 'supplementary_tables.xlsx'}")

# Note: Requires openpyxl
try:
    export_all_tables(output_dir, manifest, config, summary, run_dir)
except ImportError:
    print("Install openpyxl to export Excel: pip install openpyxl")
```

## Step 7: Statistical Comparison Table

For comparing multiple models or conditions:

```python
def create_comparison_table(run_dirs, labels):
    """Create a table comparing multiple runs."""

    rows = []
    for run_dir, label in zip(run_dirs, labels):
        summary_file = run_dir / "summary_metrics.csv"
        if summary_file.exists():
            summary = pd.read_csv(summary_file)
            auc = summary[summary["metric"] == "roc_auc"]
            acc = summary[summary["metric"] == "accuracy"]
            f1 = summary[summary["metric"] == "f1"]

            rows.append({
                "Model": label,
                "AUC": f"{auc['mean'].values[0]:.3f} ± {auc['std'].values[0]:.3f}" if not auc.empty else "N/A",
                "Accuracy": f"{acc['mean'].values[0]:.3f} ± {acc['std'].values[0]:.3f}" if not acc.empty else "N/A",
                "F1": f"{f1['mean'].values[0]:.3f} ± {f1['std'].values[0]:.3f}" if not f1.empty else "N/A",
            })

    return pd.DataFrame(rows)

# Example usage (if you have multiple runs)
# comparison = create_comparison_table(
#     [Path("run_baseline"), Path("run_smote")],
#     ["Baseline", "With SMOTE"]
# )
```

## Output Files Summary

```
outputs/06_methods_tutorial/
├── methods_paragraph.txt       # Auto-generated methods text
├── methods_detailed.md         # Detailed methods template
├── table_metrics.md            # Metrics in Markdown
├── table_metrics.tex           # Metrics in LaTeX
├── table_dataset.md            # Dataset characteristics
├── table_folds.md              # Per-fold results
└── supplementary_tables.xlsx   # All tables in Excel
```

## Citation Format

Include this in your supplementary materials:

```bibtex
@software{classiflow,
  author = {Markowitz, Alexander},
  title = {classiflow: Production-grade ML toolkit for molecular subtype classification},
  year = {2024},
  version = {0.1.0},
  url = {https://github.com/alexmarkowitz/classiflow}
}
```

## Checklist for Methods Section

- [ ] Software name and version
- [ ] Cross-validation structure (outer/inner folds)
- [ ] Model families evaluated
- [ ] Hyperparameter search space
- [ ] Class imbalance handling (SMOTE or alternative)
- [ ] Primary metric for model selection
- [ ] Random seed for reproducibility
- [ ] Data preprocessing steps
- [ ] Feature count and sample size

## Next Steps

- [Publication Checklist](../publication/checklist.md) - Complete reporting guide
- [Environment Capture](../publication/environment.md) - Documenting reproducibility
- [Citing Classiflow](../publication/citing.md) - Proper attribution
