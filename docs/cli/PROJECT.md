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
  --patient-id-col PATIENT_ID
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
