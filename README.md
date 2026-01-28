# classiflow

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Classiflow** is a CLI-first toolkit for building, evaluating, and packaging machine‑learning classifiers using **nested cross‑validation** and **hierarchical training**.

This README is intentionally short and practical. A junior developer should be able to install Classiflow, run a project, and start the UI in under 10 minutes.

---

## What you can do with Classiflow

* Run **nested cross‑validation** for unbiased model evaluation
* Train **binary**, **meta (multiclass)**, and **hierarchical** classifiers
* Enforce **patient‑level stratification** for clinical datasets
* Produce reproducible artifacts (runs, bundles, metrics, reports)

---

## Installation

Choose **one** of the following paths.

### Option A — Install to *use* Classiflow (NOT DEPLOYED AT THIS TIME)

```bash
pip install classiflow
```

Verify:

```bash
classiflow --help
```

---

### Option B — Install from source (recommended for developers)

This is the simplest and least confusing setup for local development.

```bash
git clone https://github.com/alexmarkowitz/classiflow.git
cd classiflow
pip install -e .
```

Verify:

```bash
python -c "import classiflow; print('classiflow ok')"
classiflow --help
```
---

## UI installation (optional)

The Classiflow UI is a **FastAPI + Uvicorn** service for browsing runs and artifacts.

### If installed from PyPI

```bash
pip install 'classiflow[ui]'
```

### If installed from source

```bash
pip install -e '.[ui]'
```

> **macOS / zsh note:** keep the quotes to avoid shell globbing errors.

---

## Quick start — your first Classiflow project

This is the standard workflow most users should follow.

### 1. Prepare your data

Your training data must include:

* `sample_id` (or equivalent)
* numeric feature columns
* a label column (e.g. `subtype`)
* **optional but recommended:** `patient_id` for patient‑level stratification

CSV or Parquet formats are supported.

#### Data Format Examples

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


### 2. Bootstrap a project

#### Binary classification

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Tumor Detection" \
  --mode binary \
  --label-col diagnosis \
  --out projects/
```

#### Meta (multiclass) classification

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Glioma Subtype" \
  --mode meta \
  --label-col subtype \
  --out projects/
```

#### Hierarchical classification (clinical default)

```bash
classiflow project bootstrap \
  --train-manifest data/train.csv \
  --test-manifest data/test.csv \
  --name "Brain Tumor" \
  --mode hierarchical \
  --label-col tumor_type \
  --hierarchy subtype \
  --patient-id-col patient_id \
  --out projects/
```

This creates a self‑contained project directory with configs and manifests.

---

### 3. Train nested CV models

```bash
classiflow project run-technical projects/BRAIN_TUMOR__brain_tumor
```
---

### 4. Bundle model for deployment and inference

```bash
classiflow project build-bundle projects/BRAIN_TUMOR__brain_tumor
```

This:
* selects the best model
* produces a deployable model bundle

---

### 5. Evaluate on the test set

```bash
classiflow project run-test projects/BRAIN_TUMOR__brain_tumor
```

---

## Quick start — run the Classiflow UI

1. Install UI dependencies (see above)
2. Inspect available UI commands:

```bash
classiflow ui --help
```

3. Start the server using the provided command (backed by FastAPI/Uvicorn)

The UI will expose endpoints for browsing:

* projects
* runs
* metrics
* bundles

---

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

---
## Where to go next

* Advanced workflows → `docs/`
* CLI reference → `classiflow --help`
* Project‑specific configuration → generated project folders
