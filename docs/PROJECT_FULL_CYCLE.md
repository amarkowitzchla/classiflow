# Project Full Cycle by Training Mode

This guide shows a complete project cycle for each training mode and includes
all project command options. Use it as a reference when you want reproducible,
patient-safe workflows from bootstrap through shipping.

## Common setup

Install the project extras before running feasibility or technical validation:

```bash
pip install classiflow[app,stats,viz]
```

Project structure is created under `--out` (default: `./projects`).

## Command reference (all options)

### Bootstrap

```bash
classiflow project bootstrap \
  --train-manifest PATH \
  --test-manifest PATH \
  --name NAME \
  --out projects \
  --mode auto|binary|meta|multiclass|hierarchical \
  --hierarchy L2_LABEL \
  --label-col LABEL \
  --sample-id-col SAMPLE_ID \
  --patient-col PATIENT_ID \
  --no-patient-stratified \
  --threshold metric:value \
  --copy-data copy|symlink|pointer \
  --test-id TEST_ID
```

Notes:
- `--patient-col` is optional; if set and `--no-patient-stratified` is not used,
  patient-level stratification is enabled for training workflows.
- Use `--hierarchy` only for hierarchical mode.

### Register dataset

```bash
classiflow project register-dataset \
  PROJECT_DIR \
  --type train|test \
  --manifest PATH
```

### Feasibility (stats + plots)

```bash
classiflow project run-feasibility \
  PROJECT_DIR \
  --run-id RUN_ID \
  --classes CLASS_A CLASS_B \
  --alpha 0.05 \
  --min-n 3 \
  --dunn-adjust holm \
  --top-n 30 \
  --no-legacy-csv \
  --no-legacy-xlsx \
  --no-viz \
  --fc-thresh 1.0 \
  --fc-center median \
  --label-topk 12 \
  --heatmap-topn 30 \
  --fig-dpi 160
```

### Technical validation

```bash
classiflow project run-technical \
  PROJECT_DIR \
  --run-id RUN_ID \
  --compare-smote
```

### Build bundle (final model)

```bash
classiflow project build-bundle \
  PROJECT_DIR \
  --technical-run RUN_ID \
  --run-id RUN_ID
```

### Independent test

```bash
classiflow project run-test \
  PROJECT_DIR \
  --bundle PATH \
  --final-run RUN_ID \
  --run-id RUN_ID
```

### Recommend

```bash
classiflow project recommend \
  PROJECT_DIR \
  --technical-run RUN_ID \
  --test-run RUN_ID \
  --override \
  --comment "Reason" \
  --approver "Name"
```

### Ship

```bash
classiflow project ship \
  PROJECT_DIR \
  --out PATH
```

## Mode: Binary

### Bootstrap

```bash
classiflow project bootstrap \
  --train-manifest data/train_manifest.csv \
  --test-manifest data/test_manifest.csv \
  --name "Binary Test" \
  --mode binary \
  --label-col DIAGNOSIS \
  --patient-col PATIENT_ID
```

### Full cycle

```bash
classiflow project run-feasibility projects/BINARY_TEST__binary_test
classiflow project run-technical projects/BINARY_TEST__binary_test
classiflow project build-bundle projects/BINARY_TEST__binary_test
classiflow project run-test projects/BINARY_TEST__binary_test
classiflow project recommend projects/BINARY_TEST__binary_test
classiflow project ship projects/BINARY_TEST__binary_test --out derived/ship_binary
```

## Mode: Meta (OvR + pairwise)

### Bootstrap

```bash
classiflow project bootstrap \
  --train-manifest data/train_manifest.csv \
  --test-manifest data/test_manifest.csv \
  --name "Meta Test" \
  --mode meta \
  --label-col SUBTYPE \
  --patient-col PATIENT_ID
```

### Full cycle

```bash
classiflow project run-feasibility projects/META_TEST__meta_test
classiflow project run-technical projects/META_TEST__meta_test
classiflow project build-bundle projects/META_TEST__meta_test
classiflow project run-test projects/META_TEST__meta_test
classiflow project recommend projects/META_TEST__meta_test
classiflow project ship projects/META_TEST__meta_test --out derived/ship_meta
```

## Mode: Multiclass (direct)

### Bootstrap

```bash
classiflow project bootstrap \
  --train-manifest data/train_manifest.csv \
  --test-manifest data/test_manifest.csv \
  --name "Multiclass Test" \
  --mode multiclass \
  --label-col SUBTYPE \
  --patient-col PATIENT_ID
```

To enable the torch-backed multiclass estimators on Apple Silicon, update the
project config after bootstrap:

```yaml
backend: sklearn
device: mps
```

To run only torch multiclass models (skip sklearn CPU models):

```yaml
multiclass:
  estimator_mode: torch_only
```

### Full cycle

```bash
classiflow project run-feasibility projects/MULTICLASS_TEST__multiclass_test
classiflow project run-technical projects/MULTICLASS_TEST__multiclass_test
classiflow project build-bundle projects/MULTICLASS_TEST__multiclass_test
classiflow project run-test projects/MULTICLASS_TEST__multiclass_test
classiflow project recommend projects/MULTICLASS_TEST__multiclass_test
classiflow project ship projects/MULTICLASS_TEST__multiclass_test --out derived/ship_multiclass
```

## Mode: Hierarchical

### Bootstrap

```bash
classiflow project bootstrap \
  --train-manifest data/hierarchical_train.csv \
  --test-manifest data/hierarchical_test.csv \
  --name "Hier Test" \
  --mode hierarchical \
  --label-col TUMOR_TYPE \
  --hierarchy SUBTYPE \
  --patient-col PATIENT_ID
```

### Full cycle

```bash
classiflow project run-feasibility projects/HIER_TEST__hier_test
classiflow project run-technical projects/HIER_TEST__hier_test
classiflow project build-bundle projects/HIER_TEST__hier_test
classiflow project run-test projects/HIER_TEST__hier_test
classiflow project recommend projects/HIER_TEST__hier_test
classiflow project ship projects/HIER_TEST__hier_test --out derived/ship_hier
```

## Configuration checkpoints

After bootstrap, verify `project.yaml` has:

```yaml
task:
  mode: meta
  patient_stratified: true
  hierarchy_path: null
key_columns:
  label: SUBTYPE
  patient_id: PATIENT_ID
backend: sklearn
device: auto
model_set: default
torch_dtype: float32
torch_num_workers: 0
```

For binary/meta GPU runs, set:

```yaml
backend: torch
device: mps
model_set: torch_basic
```

For hierarchical:

```yaml
task:
  mode: hierarchical
  patient_stratified: true
  hierarchy_path: SUBTYPE
key_columns:
  label: TUMOR_TYPE
  patient_id: PATIENT_ID
```
