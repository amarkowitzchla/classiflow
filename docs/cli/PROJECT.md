# Project CLI

## Commands

- `classiflow project init --name NAME`
- `classiflow project bootstrap --train-manifest PATH --test-manifest PATH --name NAME`
- `classiflow project register-dataset --type train|test`
- `classiflow project run-feasibility`
- `classiflow project run-technical`
- `classiflow project build-bundle`
- `classiflow project run-test`
- `classiflow project recommend`
- `classiflow project ship`

Use `--mode multiclass` in `bootstrap` for direct multiclass training.
Provide `--patient-col` (or `--patient-id-col`) with `--no-patient-stratified` unset to enable patient-level stratification for binary, meta, multiclass, or hierarchical modes.

For end-to-end examples with all options per training mode, see `docs/PROJECT_FULL_CYCLE.md`.

## Backend Options

Project YAML supports backend settings for `train-binary` and `train-meta`:

```yaml
backend: sklearn
device: auto
model_set: default
torch_dtype: float32
torch_num_workers: 0
```

Example torch configuration:

```yaml
backend: torch
device: mps
model_set: torch_basic
```

For binary/meta GPU acceleration, set `backend: torch` and a GPU device
(`mps` on Apple Silicon). `backend: sklearn` always runs CPU estimators.

For multiclass training, keep `backend: sklearn` and set `device: mps` (or `cuda`) to
enable the torch-backed multiclass estimators added to the nested CV search:

```yaml
backend: sklearn
device: mps
```

To force multiclass runs to use only torch models:

```yaml
multiclass:
  estimator_mode: torch_only
```

## Multiclass Technical Validation

Multiclass technical validation defaults to stratified group splits (patient-safe) when `patient_id` is set.
You can tune the default logistic regression baseline in `project.yaml`:

```yaml
multiclass:
  group_stratify: true  # set false to use group-only (non-stratified) splits
  logreg:
    solver: saga
    multi_class: auto
    penalty: l2
    max_iter: 5000
    tol: 1.0e-3
    C: 1.0
    class_weight: balanced
    n_jobs: -1
```

`multi_class: auto` selects multinomial for multiclass problems with saga. If you still hit convergence issues, increase `logreg.max_iter` or relax `logreg.tol`.

## Hierarchical Mode

Required columns:
- Level-1 label column (e.g., `TUMOR_TYPE`)
- Level-2 label column (e.g., `SUBTYPE`)

Optional:
- Patient ID column for patient-level stratification

Examples:

```bash
classiflow project bootstrap \
  --train-manifest data/hierarchical_train.csv \
  --name "Hier Test" \
  --mode hierarchical \
  --label-col TUMOR_TYPE \
  --hierarchy SUBTYPE \
  --patient-col PATIENT_ID
```

Disable patient stratification:

```bash
classiflow project bootstrap \
  --train-manifest data/hierarchical_train.csv \
  --name "Hier Test" \
  --mode hierarchical \
  --label-col TUMOR_TYPE \
  --hierarchy SUBTYPE \
  --no-patient-stratified
```

## Shipping for Deployment

Copy the exact bundle and metadata to a deployment directory:

```bash
classiflow project ship projects/MRS_MEDULLO__mrs_medullo --out /tmp/deploy_bundle
```

## Examples

```bash
classiflow project init --name "Glioma Subtype" --test-id GLIOMA01
classiflow project register-dataset projects/GLIOMA01__glioma_subtype --type train
classiflow project run-feasibility projects/GLIOMA01__glioma_subtype
classiflow project run-technical projects/GLIOMA01__glioma_subtype
classiflow project build-bundle projects/GLIOMA01__glioma_subtype
classiflow project run-test projects/GLIOMA01__glioma_subtype
classiflow project recommend projects/GLIOMA01__glioma_subtype
```
