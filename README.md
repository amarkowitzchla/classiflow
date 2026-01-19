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

Install the latest release from PyPI:

```bash
pip install classiflow
```

Optional dependency bundles follow the names declared in `pyproject.toml`. Because zsh treats `[...]` as globbing, wrap each requirement in quotes (or escape the brackets) when using the default shell:

```bash
pip install 'classiflow[app]'     # Streamlit UI helpers (app, Plotly, Excel exports)
pip install 'classiflow[ui]'      # FastAPI + Uvicorn services used by the admin dashboard
pip install 'classiflow[viz]'     # UMAP + Plotly visualization tools
pip install 'classiflow[stats]'   # Statistical analysis helpers
pip install 'classiflow[parquet]' # Parquet I/O support (pyarrow)
pip install 'classiflow[dev]'     # Testing, linting, and packaging tools
pip install 'classiflow[docs]'    # MkDocs-based documentation toolchain
pip install 'classiflow[all]'     # Everything (app, ui, viz, stats, parquet, dev, docs)
```

Before you run `classiflow project run-feasibility` or `classiflow project run-technical`, install the `app`, `stats`, and `viz` extras so the Excel reports, pairwise statistics, and UMAP plots can find `xlsxwriter`, `statsmodels`, `scikit-posthocs`, and `umap-learn`.

From a local checkout, install the package (optionally editable) instead of relying on the prebuilt wheel:

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

## Data Input

Classiflow supports multiple data formats for flexibility and performance:

| Format | Extension | Description | Best For |
| --- | --- | --- | --- |
| **Parquet** (recommended) | `.parquet` | Columnar binary format with schema | Large datasets, faster I/O |
| CSV | `.csv` | Plain-text comma-separated values | Compatibility, small datasets |
| Parquet Dataset | `directory/` | Directory with chunked `.parquet` files | Very large datasets, partitioned data |

### Using Different Formats

All training and inference commands accept the `--data` option (preferred) or `--data-csv` (deprecated):

```bash
# Single Parquet file (recommended)
classiflow train-meta --data data.parquet --label-col subtype

# Parquet dataset directory (chunked files)
classiflow train-meta --data data_parquet_dir/ --label-col subtype

# Legacy CSV (deprecated but supported)
classiflow train-meta --data-csv data.csv --label-col subtype
```

For Parquet support, install the optional dependency:

```bash
pip install classiflow[parquet]
# or
pip install pyarrow
```

### Creating Parquet Files

Convert existing CSV files to Parquet for better performance:

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", index=False)
```

For large datasets, create chunked parquet directories:

```python
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

df = pd.read_csv("large_data.csv")

# Split into chunks of 1000 rows each
chunk_size = 1000
for i, start in enumerate(range(0, len(df), chunk_size)):
    chunk = df.iloc[start:start + chunk_size]
    chunk.to_parquet(f"data_dir/part-{i:03d}.parquet", index=False)
```

---

## Quick Start: Project Workflow (Bootstrap → Ship)

The project workflow provides a structured pipeline for developing clinical ML tests. It handles dataset registration, validation, model training, evaluation, and deployment.

### Data Format Examples

**Binary classification** (two classes):

```csv
sample_id,feature_1,feature_2,feature_3,diagnosis
S001,0.52,1.23,0.87,Tumor
S002,0.31,0.95,1.12,Normal
S003,0.67,1.45,0.93,Tumor
```

**Multiclass / meta classification** (3+ classes):

```csv
sample_id,feature_1,feature_2,feature_3,subtype
S001,0.52,1.23,0.87,TypeA
S002,0.31,0.95,1.12,TypeB
S003,0.67,1.45,0.93,TypeC
S004,0.44,1.10,0.88,TypeA
```

**Hierarchical classification** (two-level labels with optional patient grouping):

```csv
sample_id,patient_id,feature_1,feature_2,tumor_type,subtype
S001,P001,0.52,1.23,Glioma,Astrocytoma
S002,P001,0.31,0.95,Glioma,Oligodendroglioma
S003,P002,0.67,1.45,Meningioma,Grade_I
S004,P003,0.44,1.10,Glioma,GBM
```

---

### Step 1: Bootstrap the Project

Choose the appropriate mode based on your classification task.

**Binary mode** (two-class classification):

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Tumor Detection" \
  --mode binary \
  --label-col diagnosis
```

**Meta mode** (multiclass via OvR + pairwise binary tasks):

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Glioma Subtype" \
  --mode meta \
  --label-col subtype
```

**Hierarchical mode** (two-level classification with patient stratification):

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Brain Tumor" \
  --mode hierarchical \
  --label-col tumor_type \
  --hierarchy subtype \
  --patient-id-col patient_id
```

**Hierarchical mode** (sample-level splits, no patient grouping):

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --name "Brain Tumor" \
  --mode hierarchical \
  --label-col tumor_type \
  --hierarchy subtype \
  --no-patient-stratified
```

Bootstrap options:
- `--mode`: `auto` | `binary` | `meta` | `hierarchical` (default: `auto`)
- `--test-manifest`: Optional; register later with `classiflow project register-dataset`
- `--copy-data`: `copy` | `symlink` | `pointer` (default: `pointer`)
- `--threshold`: Override metric thresholds, e.g., `--threshold auc:0.85`

---

### Step 2: Run Technical Validation (Optional)

Validate data quality and configuration:

```bash
classiflow project run-technical projects/TUMOR_DETECTION__tumor_detection
```

---

### Step 3: Run Feasibility Analysis (Optional)

Generate exploratory statistics and visualizations:

```bash
classiflow project run-feasibility projects/TUMOR_DETECTION__tumor_detection
```

---

### Step 4: Build the Model Bundle

Train the final model and package it:

```bash
classiflow project build-bundle projects/TUMOR_DETECTION__tumor_detection
```

This runs the appropriate training command (`train-binary`, `train-meta`, or `train-hierarchical`) based on the project mode and creates a deployable bundle.

---

### Step 5: Run Independent Test Evaluation

Evaluate performance on the held-out test set:

```bash
classiflow project run-test projects/TUMOR_DETECTION__tumor_detection
```

---

### Step 6: Generate Promotion Recommendation

Evaluate promotion gates and emit a go/no-go recommendation:

```bash
classiflow project recommend projects/TUMOR_DETECTION__tumor_detection
```

Override failed gates (with justification):

```bash
classiflow project recommend projects/TUMOR_DETECTION__tumor_detection \
  --override \
  --comment "Approved with known limitation" \
  --approver "user@example.com"
```

---

### Step 7: Ship for Deployment

Export the bundle and metadata for production deployment:

```bash
classiflow project ship projects/TUMOR_DETECTION__tumor_detection \
  --out deploy/tumor_detection_v1
```

The ship directory includes:
- `model_bundle.zip` — deployable model artifact
- `run.json` — training run metadata
- `lineage.json` — data and model lineage
- `decision.yaml` — promotion decision record
- `ship_manifest.yaml` — deployment index

---

### Step 8: Validate and Use the Shipped Bundle

Inspect and validate the bundle:

```bash
classiflow bundle validate deploy/tumor_detection_v1/model_bundle.zip
classiflow bundle inspect deploy/tumor_detection_v1/model_bundle.zip --verbose
```

Run inference on new samples:

```bash
classiflow infer \
  --bundle deploy/tumor_detection_v1/model_bundle.zip \
  --data-csv data/new_samples.csv \
  --outdir results/
```

Predictions now include calibrated `y_prob` plus `y_score_raw` (uncalibrated) and calibration metadata columns; inference also exports a `calibration_curve.csv` plus per-run Brier/ECE/logloss entries in the metrics workbook.

---

## Standalone Training Commands

For direct training without the project workflow, use these commands:

### train-binary

Nested CV binary classification with SMOTE support:

```bash
classiflow train-binary \
  --data data/features.parquet \
  --label-col diagnosis \
  --pos-label "Tumor" \
  --smote both \
  --outer-folds 5 \
  --inner-splits 5 \
  --inner-repeats 2 \
  --outdir derived/binary \
  --verbose
```

Options:
- `--data`: Path to data file (.csv, .parquet) or directory (parquet dataset)
- `--smote`: `off` | `on` | `both` (default: `off`)
- `--pos-label`: Specify positive class (default: minority class)
- `--max-iter`: Linear solver iterations (default: 10000)

### train-meta

Multiclass via combined OvR + pairwise binary tasks:

```bash
classiflow train-meta \
  --data data/features.parquet \
  --label-col subtype \
  --smote both \
  --outer-folds 5 \
  --inner-splits 5 \
  --outdir derived/meta \
  --verbose
```

With custom composite tasks from JSON:

```bash
classiflow train-meta \
  --data data/features.parquet \
  --label-col subtype \
  --tasks-json tasks.json \
  --tasks-only \
  --smote both \
  --outdir derived/meta
```

Options:
- `--data`: Path to data file (.csv, .parquet) or directory (parquet dataset)
- `--classes`: Subset/order of classes to include
- `--tasks-json`: Custom task definitions
- `--tasks-only`: Skip auto OvR/pairwise, use only JSON tasks
- `--calibrate-meta/--no-calibrate-meta`: Enable or disable meta-classifier probability calibration (default: enabled)
- `--calibration-method`: Calibration strategy (`sigmoid` (default) or `isotonic`)
- `--calibration-cv`: Number of folds for the cross-validated calibrator
- `--calibration-bins`: Number of bins used when computing calibration curves / ECE

Meta training now records calibrated probabilities (see new `y_prob` columns) and exports fold-level `calibration_curve.csv` + `calibration_summary.json`, while the reports aggregate Brier/ECE/logloss for the calibrated meta predictions.
### train-hierarchical

Two-level classification with optional patient stratification:

```bash
classiflow train-hierarchical \
  --data data/features.parquet \
  --patient-col patient_id \
  --label-l1 tumor_type \
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

Single-level (flat multiclass) without L2:

```bash
classiflow train-hierarchical \
  --data data/features.parquet \
  --label-l1 diagnosis \
  --device auto \
  --outdir derived/flat_multiclass
```

Options:
- `--data`: Path to data file (.csv, .parquet) or directory (parquet dataset)
- `--patient-col`: Enable patient-level stratification (prevents leakage)
- `--label-l2`: Enable hierarchical L1→L2 routing
- `--device`: `auto` | `cpu` | `cuda` | `mps`
- `--l2-classes`: Subset of L2 classes to include
- `--min-l2-classes-per-branch`: Minimum L2 classes per branch (default: 2)

---

## Additional Utilities

**Statistical analysis:**

```bash
classiflow stats run --data data/features.parquet --label-col diagnosis --outdir derived/stats
classiflow stats viz --data data/features.parquet --label-col diagnosis --stats-dir derived/stats
```

**Inference:**

```bash
# Run inference on new data
classiflow infer \
  --data data/test.parquet \
  --run-dir derived/meta/fold1 \
  --outdir results/

# Or use a model bundle
classiflow infer \
  --data data/test.parquet \
  --bundle models/model.zip \
  --outdir results/
```

**Bundle management:**

```bash
classiflow bundle create --run-dir derived/binary --out bundle.zip --all-folds
classiflow bundle inspect bundle.zip --verbose
classiflow bundle validate bundle.zip
```

**Migrate legacy runs:**

```bash
classiflow migrate run derived/binary --data-csv data/features.csv
classiflow migrate batch archived_runs --pattern "derived_*" --dry-run
```

## Next Steps

- See [PROJECT_QUICKSTART.md](docs/PROJECT_QUICKSTART.md) for detailed project workflow documentation
- Consult [HIERARCHICAL_TRAINING.md](HIERARCHICAL_TRAINING.md) for advanced hierarchical mode options, GPU tuning, and troubleshooting
- Run `classiflow stats umap` via `scripts/umap_plot.py` for dimensionality reduction visuals
- Explore `src/classiflow/cli` to extend or script these workflows
