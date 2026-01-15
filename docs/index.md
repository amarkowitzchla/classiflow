# Classiflow

**Production-grade ML toolkit for molecular subtype classification with nested cross-validation and publication-ready reporting.**

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started in Minutes**

    ---

    Install classiflow with pip and run your first classification pipeline in under 10 lines of code.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Comprehensive Tutorials**

    ---

    Step-by-step guides covering binary classification, multiclass, imbalanced data, and publication figures.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-api:{ .lg .middle } **Full API Reference**

    ---

    Complete documentation of all modules, classes, and functions with examples.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-file-document-edit:{ .lg .middle } **Publication Ready**

    ---

    Generate methods paragraphs, figures, and tables suitable for peer-reviewed manuscripts.

    [:octicons-arrow-right-24: Publication Guide](publication/index.md)

</div>

## What is Classiflow?

Classiflow is a Python toolkit designed for researchers who need rigorous, reproducible machine learning classification pipelines. It was built with molecular subtyping in mind but applies to any tabular classification problem where you need:

- **Unbiased evaluation** via nested cross-validation
- **Proper handling of class imbalance** with SMOTE inside cross-validation folds
- **Publication-ready outputs** including metrics, figures, and methods text
- **Complete reproducibility** with run manifests, data hashing, and environment capture

## Key Features

### Nested Cross-Validation

Nested CV separates hyperparameter tuning from performance estimation, preventing optimistic bias in reported metrics. Classiflow handles this automatically:

```python
from classiflow import train_binary_task, TrainConfig

config = TrainConfig(
    data_csv="features.csv",
    label_col="diagnosis",
    pos_label="Disease",
    outer_folds=5,      # Performance estimation
    inner_splits=5,     # Hyperparameter tuning
    inner_repeats=2,    # Repeated inner CV for stability
)

results = train_binary_task(config)
```

### Three Training Paradigms

| Mode | Use Case | Command |
|------|----------|---------|
| **Binary** | Single two-class task | `classiflow train-binary` |
| **Meta** | Multiclass via binary task ensemble | `classiflow train-meta` |
| **Hierarchical** | L1 → branch-specific L2 with patient stratification | `classiflow train-hierarchical` |

### Adaptive SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is applied correctly—inside each training fold—to prevent data leakage:

```python
config = TrainConfig(
    data_csv="features.csv",
    label_col="diagnosis",
    smote_mode="both",  # Compare with and without SMOTE
    smote_k_neighbors=5,
)
```

### Complete Lineage Tracking

Every training run generates a `run.json` manifest with:

- SHA256 hash of training data
- Git commit hash
- Python version and hostname
- Complete configuration
- Feature list and summaries
- Best model per task

### Publication Outputs

Generate figures and tables ready for manuscripts:

- ROC curves with AUC confidence intervals
- Precision-recall curves
- Confusion matrices
- Calibration plots
- Feature importance rankings
- Excel workbooks with all metrics

## Quick Example

```python
from classiflow import train_binary_task, TrainConfig
from classiflow.inference import run_inference, InferenceConfig

# Train with nested CV
train_config = TrainConfig(
    data_csv="data/features.csv",
    label_col="diagnosis",
    pos_label="Malignant",
    outer_folds=5,
    smote_mode="both",
    outdir="derived/binary_run",
)
results = train_binary_task(train_config)

# Run inference on new data
infer_config = InferenceConfig(
    run_dir="derived/binary_run",
    data_csv="data/new_samples.csv",
    output_dir="derived/inference",
    label_col="diagnosis",  # Optional: compute metrics
)
predictions = run_inference(infer_config)
```

## When to Use Classiflow

Classiflow is ideal when you need:

- **Rigorous evaluation** for peer-reviewed publications
- **Reproducible pipelines** that colleagues can verify
- **Imbalanced data handling** done correctly (SMOTE inside folds)
- **Multiple training modes** (binary, multiclass, hierarchical)
- **Patient-level stratification** to prevent data leakage in medical imaging
- **Publication-ready artifacts** (figures, tables, methods text)

## Installation

```bash
pip install classiflow
```

Or with optional dependencies:

```bash
pip install classiflow[all]  # Everything
pip install classiflow[stats]  # Statistical analysis
pip install classiflow[app]    # Streamlit UI
```

See the [Installation Guide](getting-started/installation.md) for details.

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Install**

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-play:{ .lg .middle } **Run Your First Pipeline**

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-school:{ .lg .middle } **Learn the Concepts**

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   :material-github:{ .lg .middle } **Contribute**

    [:octicons-arrow-right-24: GitHub](https://github.com/alexmarkowitz/classiflow)

</div>

## Citation

If you use classiflow in your research, please cite:

```bibtex
@software{markowitz2024classiflow,
  author = {Markowitz, Alexander},
  title = {classiflow: Production-grade ML toolkit for molecular subtype classification},
  year = {2024},
  url = {https://github.com/alexmarkowitz/classiflow},
  version = {0.1.0}
}
```

## License

Classiflow is released under the [MIT License](https://opensource.org/licenses/MIT).
