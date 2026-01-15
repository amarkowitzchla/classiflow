# classiflow

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade toolkit for molecular subtype classification, built around nested cross-validation and hierarchical training.

## Key Capabilities

- **Nested cross-validation** for unbiased hyperparameter tuning and evaluation.
- **Binary, meta, and hierarchical task orchestration** with patient-level stratification to prevent leakage.
- **PyTorch MLP + GPU support** (CUDA/MPS) with early stopping and adaptive SMOTE.
- **CLI-first workflows** for training, inference, statistical analysis, model bundling, and migration.
- **Artifact lineage** with run manifests, hashes, and portable bundles for reproducibility.

## Installation

```bash
pip install classiflow
```

Optionally install extras:

```bash
pip install classiflow[app]    # Streamlit UI
pip install classiflow[viz]    # Visualization helpers
pip install classiflow[stats]  # Statistical modules
pip install classiflow[dev]    # Developer tools
pip install classiflow[all]    # Everything
```

For development:

```bash
git clone https://github.com/alexmarkowitz/classiflow.git
cd classiflow
pip install -e ".[dev]"
```

## CLI Overview

| Command | Purpose |
| --- | --- |
| `classiflow train-binary` | Nested CV binary task with SMOTE, inner repeats, and linear model regularization. |
| `classiflow train-meta` | Train meta-classifier by combining OvR, pairwise, and optional composite tasks from JSON. |
| `classiflow train-hierarchical` | Patient-stratified L1 (and optional L2) nested CV using a PyTorch MLP. |
| `classiflow infer` | Inference suite for binary/meta models; supports inference bundles, strict/lenient feature alignment, Excel/plots. |
| `classiflow infer-hierarchical` | Single-command inference for hierarchical models (auto L1→L2 routing plus uncertainty). |
| `classiflow summarize` | Aggregate fold metrics into summary reports (pending implementation placeholder). |
| `classiflow export-best` | Export top-performing binary/meta tasks into spreadsheets. |
| `classiflow stats run` | Run statistical analyses (normality, pairwise tests, publication-ready workbook). |
| `classiflow stats viz` | Generate plots (boxplots, volcano, heatmaps, UMAP/t-SNE) from stats results. |
| `classiflow stats umap` | Run UMAP visualization (script stub referencing `scripts/umap_plot.py`). |
| `classiflow bundle create/inspect/validate` | Package training artifacts into ZIP, inspect metadata, and validate completeness. |
| `classiflow migrate run/batch` | Migrate legacy run directories to the new lineage (`run.json`) format. |

## Quick Start: Binary Workflow

1. **Prepare your CSV** with feature columns plus a label column (`diagnosis`, `subtype`, etc.). Include non-numeric columns as long as you exclude them from the feature set.
2. **Train the binary task**:

```bash
classiflow train-binary \
  --data-csv data/features.csv \
  --label-col diagnosis \
  --pos-label "PositiveClass" \
  --smote both \
  --outer-folds 5 \
  --inner-splits 5 \
  --inner-repeats 2 \
  --outdir derived/binary \
  --verbose
```

   - `--smote` defaults to `off`; use `on`/`both` to synthesize minority-class samples.
   - `--max-iter` controls the linear solver convergence for logistic/ridge classifiers.
3. **Summarize or export** training metrics for quick reporting:

```bash
classiflow summarize derived/binary
classiflow export-best derived/binary
```

4. **Create a portable bundle** for deployment or sharing:

```bash
classiflow bundle create \
  --run-dir derived/binary \
  --out artifacts/binary_bundle.zip \
  --all-folds \
  --description "Binary nested CV v0.2"
classiflow bundle inspect artifacts/binary_bundle.zip
classiflow bundle validate artifacts/binary_bundle.zip
```

5. **Run inference** on new samples:

```bash
classiflow infer \
  --data-csv data/new_samples.csv \
  --run-dir derived/binary \
  --outdir derived/binary_infer \
  --label-col diagnosis \
  --lenient \
  --fill-strategy median \
  --device cpu
```

   - Use `--bundle` instead of `--run-dir` when deploying from a ZIP.
   - `--no-plots` / `--no-excel` skip optional outputs when needed.

6. **Surface statistical insights**:

```bash
classiflow stats run --data-csv data/features.csv --label-col diagnosis --outdir derived/stats
classiflow stats viz --data-csv data/features.csv --label-col diagnosis --stats-dir derived/stats
```

7. **Migrate legacy runs** if you have `run_manifest.json` files:

```bash
classiflow migrate run derived/binary --data-csv data/features.csv
classiflow migrate batch archived_runs --pattern "derived_*" --dry-run
```

## Quick Start: Hierarchical Workflow

Hierarchical mode routes patients through Level-1 → branch-specific Level-2 classifiers while respecting patient-level stratification. Refer to [`HIERARCHICAL_TRAINING.md`](HIERARCHICAL_TRAINING.md) for the full guide.

1. **Structure your CSV** with `patient_id`/`svs_id`, `label_l1`, optional `label_l2`, and numeric feature columns.
2. **Train the hierarchical model**:

```bash
classiflow train-hierarchical \
  --data-csv data/features.csv \
  --patient-col patient_id \
  --label-l1 diagnosis \
  --label-l2 subtype \
  --device auto \
  --use-smote \
  --outer-folds 5 \
  --inner-splits 3 \
  --mlp-epochs 100 \
  --early-stopping-patience 10 \
  --outdir derived/hierarchical \
  --verbose 2
```

   - Omit `--label-l2` for flat multiclass training; include it to enable branch models.
   - Control L2 routing via `--l2-classes` and `--min-l2-classes-per-branch`.
   - `--device` accepts `auto`, `cpu`, `cuda`, or `mps`.

3. **Inspect training outputs**: each fold directory contains scalers, encoders, configs, and `metrics_outer_eval.xlsx`.
4. **Bundle hierarchical outputs** the same way as binary models (use `mlsubtype bundle`).
5. **Run hierarchical inference**:

```bash
classiflow infer-hierarchical \
  --data-csv data/new_tiles.csv \
  --model-dir derived/hierarchical \
  --fold 1 \
  --device cuda \
  --outfile derived/hierarchical_predictions.csv \
  --include-proba
```

   - Outputs include `l1_class`, `l2_class`, and optional probabilities/uncertainty.
6. **Use `mlsubtype infer`** when you want the full inference stack (Excel reports, ROC plots, metrics) from a bundle or run directory.

## Next Steps

- Explore `ENVIRONMENT.md`, `MANIFEST.in`, and `src/classiflow/cli` if you want to extend or script these workflows.
- Run `classiflow stats umap` via the standalone script in `scripts/umap_plot.py` for dimensionality reduction visuals.
- Consult `HIERARCHICAL_TRAINING.md` for advanced sampling, GPU tuning, and troubleshooting.

