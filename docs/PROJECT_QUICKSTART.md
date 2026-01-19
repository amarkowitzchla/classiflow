# Quickstart: Clinical Test Project

This guide walks through an end-to-end clinical test workflow using the project module.

## 1) Bootstrap a project

```bash
classiflow project bootstrap \
  --train-manifest data/train_manifest.csv \
  --test-manifest data/test_manifest.csv \
  --name "Glioma Subtype" \
  --mode meta
  --label-col SUBTYPE
```

If you do not have an independent test set yet, omit `--test-manifest` and register it later.

For hierarchical workflows, pass the level-2 label column via `--hierarchy` and update `labels.yaml` as needed.

## Hierarchical Setup

The manifest must include a level-1 label column and a level-2 label column. Optionally include a patient ID
column if you want patient-level stratification.

With patient stratification:

```bash
classiflow project bootstrap \
  --train-manifest data/hierarchical_train.csv \
  --name "Hier Test" \
  --mode hierarchical \
  --label-col TUMOR_TYPE \
  --hierarchy SUBTYPE \
  --patient-id-col PATIENT_ID
```

Without patient stratification (sample-level splits):

```bash
classiflow project bootstrap \
  --train-manifest data/hierarchical_train.csv \
  --name "Hier Test" \
  --mode hierarchical \
  --label-col TUMOR_TYPE \
  --hierarchy SUBTYPE \
  --no-patient-stratified
```

After bootstrap, verify `project.yaml` has:

```yaml
task:
  mode: hierarchical
  patient_stratified: true
  hierarchy_path: SUBTYPE
key_columns:
  label: TUMOR_TYPE
  patient_id: PATIENT_ID
```

Before you run the feasibility or technical validation steps, install the extras that provide the required Excel and stats tooling:

```bash
pip install classiflow[app,stats,viz]
```
`app` supplies `xlsxwriter`/`openpyxl` for the validation spreadsheets, `stats` gives you `statsmodels` and `scikit-posthocs`, and `viz` brings in `umap-learn` now that `scikit-learn` allows versions up to 2.0.0.

## 2) Run feasibility (optional)

```bash
classiflow project run-feasibility projects/GLIOMA_SUBTYPE__glioma_subtype
```

## 3) Run technical validation

```bash
classiflow project run-technical projects/GLIOMA_SUBTYPE__glioma_subtype
```

## 4) Train final model + bundle

```bash
classiflow project build-bundle projects/GLIOMA_SUBTYPE__glioma_subtype
```

## 5) Run independent test evaluation

```bash
classiflow project run-test projects/GLIOMA_SUBTYPE__glioma_subtype
```

## 6) Promotion recommendation

```bash
classiflow project recommend projects/GLIOMA_SUBTYPE__glioma_subtype
```

## 7) Ship for deployment

```bash
classiflow project ship projects/HIER_TEST__hier_test --out derived/test_ship
```

The ship directory includes the bundle (`model_bundle.zip`) plus `run.json`,
`lineage.json`, `decision.yaml`, and a `ship_manifest.yaml` index.

## 7) Install in a pipeline (using classiflow)

Validate and inspect the shipped bundle:

```bash
classiflow bundle validate derived/test_ship/model_bundle.zip
classiflow bundle inspect derived/test_ship/model_bundle.zip --verbose
```

Run inference directly from the bundle:

```bash
classiflow infer \
  --bundle derived/test_ship/model_bundle.zip \
  --data-csv data/new_samples.csv \
  --outdir results/
```

## 8) Generate bundle documentation

Use the bundle inspector to create a portable, human-readable summary:

```bash
classiflow bundle inspect derived/test_ship/model_bundle.zip --verbose > derived/test_ship/bundle_report.txt
```

## Notes

- Edit `project.yaml` to match your manifest column names.
- For multiclass + MPS acceleration, set `device: mps` in `project.yaml` (keep `backend: sklearn`).
- To run only torch multiclass models, set `multiclass.estimator_mode: torch_only`.
- For binary/meta GPU acceleration, set `backend: torch`, `device: mps`, and `model_set: torch_basic`.
- If you must run only when MPS/CUDA is available, set `require_torch_device: true`.
- Configure thresholds in `registry/thresholds.yaml`.
- Calibration gating defaults live under `promotion.calibration` (Brier/ECE limits).
- Artifacts are stored under `runs/` and `promotion/`.

## UI installation and static files

The UI backend requires the optional UI dependencies (FastAPI + Uvicorn). If you install from a wheel,
include the extras:

```bash
python3 -m pip install "classiflow-*.whl[ui]"
```

The UI frontend is a separate React build. If you see `GET / 404` it means the backend is running in API-only
mode without static files. Build the frontend and point the server at it:

```bash
cd classiflow-ui
npm install
npm run build
classiflow ui serve --projects-root ./projects --static-dir ./classiflow-ui/dist
```

Alternatively, use the Docker UI image (`Dockerfile.ui`) which builds and serves the static assets for you.
