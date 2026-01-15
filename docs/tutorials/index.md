# Tutorials

Step-by-step guides for common classiflow workflows.

These tutorials are designed to be followed in order, building from simple binary classification to advanced publication-ready outputs.

## Learning Path

<div class="grid cards" markdown>

-   :material-numeric-1-circle:{ .lg .middle } **Binary Classification**

    ---

    Train a two-class classifier with nested cross-validation. Covers data preparation, model training, and results interpretation.

    [:octicons-arrow-right-24: Start Tutorial](binary-classification.md)

-   :material-numeric-2-circle:{ .lg .middle } **Multiclass Classification**

    ---

    Handle three or more classes using meta-classifiers. Learn about OvR, pairwise, and composite task strategies.

    [:octicons-arrow-right-24: Start Tutorial](multiclass-classification.md)

-   :material-numeric-3-circle:{ .lg .middle } **Imbalanced Data & SMOTE**

    ---

    Properly handle class imbalance with SMOTE applied inside CV folds. Compare results with and without oversampling.

    [:octicons-arrow-right-24: Start Tutorial](imbalanced-data.md)

-   :material-numeric-4-circle:{ .lg .middle } **Cross-Validation & Reproducibility**

    ---

    Deep dive into nested CV, random seeds, and ensuring reproducible results.

    [:octicons-arrow-right-24: Start Tutorial](cross-validation.md)

-   :material-numeric-5-circle:{ .lg .middle } **Publication Figures**

    ---

    Generate ROC curves, confusion matrices, calibration plots, and feature importance figures for manuscripts.

    [:octicons-arrow-right-24: Start Tutorial](publication-figures.md)

-   :material-numeric-6-circle:{ .lg .middle } **Methods Text & Tables**

    ---

    Create methods paragraphs and summary tables suitable for peer-reviewed publications.

    [:octicons-arrow-right-24: Start Tutorial](methods-tables.md)

</div>

## Prerequisites

Before starting these tutorials, ensure you have:

1. **Installed classiflow** with all extras:
   ```bash
   pip install classiflow[all]
   ```

2. **Sample data** - Tutorials use scikit-learn's built-in datasets (automatically available)

3. **Basic Python knowledge** - Familiarity with pandas, numpy, and scikit-learn helps but isn't required

## Tutorial Data

All tutorials use synthetic or built-in datasets to ensure reproducibility without external downloads:

- **Binary tutorials**: Breast cancer dataset (`sklearn.datasets.load_breast_cancer`)
- **Multiclass tutorials**: Iris dataset (`sklearn.datasets.load_iris`)
- **Imbalanced tutorials**: Synthetic imbalanced data via `make_classification`

## Running Tutorial Code

Each tutorial includes complete, runnable code. You can:

1. **Copy-paste** code blocks directly into Python or Jupyter
2. **Run the example scripts** in `examples/` directory
3. **Use the Streamlit app** for interactive exploration

## Expected Outputs

Tutorials produce outputs in `./outputs/<tutorial_name>/`:

```
outputs/
├── 01_quickstart_binary/
│   ├── run.json
│   ├── summary_metrics.csv
│   └── fold_*/
├── 02_multiclass/
│   └── ...
└── ...
```

## Getting Help

- Check the [FAQ](../faq.md) for common questions
- See [Concepts](../concepts/index.md) for theoretical background
- Review [API Reference](../api/index.md) for detailed parameter documentation
