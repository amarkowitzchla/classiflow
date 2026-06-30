# inference

## Objective
- Provide end-to-end inference for trained runs and bundles across run types:
  - binary pipelines, meta-classifier, multiclass models, and hierarchical models.
- Enforce a stable **feature alignment** contract between training and inference.
- Produce reviewer-friendly outputs (predictions CSV, metrics workbook, plots) with traceable warnings.

## Public Interfaces
- Main API:
  - `run_inference(config: InferenceConfig) -> dict` in `src/classiflow/inference/api.py`
- Configs:
  - `InferenceConfig` in `src/classiflow/inference/config.py`
  - `RunManifest` in `src/classiflow/inference/config.py` (inference-side manifest representation)
- High-level helpers (exported via `src/classiflow/inference/__init__.py`):
  - `HierarchicalInference` in `src/classiflow/inference/hierarchical.py`
- CLI:
  - `classiflow infer ...` in `src/classiflow/cli/main.py`
  - `classiflow infer-hierarchical ...` in `src/classiflow/cli/main.py`

Sub-docs:
- `docs/change-management/inference/run_inference.md`
- `docs/change-management/inference/artifact_loading.md`
- `docs/change-management/inference/predictors.md`

## Inputs
- Model source:
  - a run directory (`--run-dir`) that contains artifacts, possibly including `run.json`
  - or a bundle (`--bundle`) which is extracted and mapped to a fold directory
- Data source:
  - `--data` preferred (CSV/Parquet/dataset dir), legacy `--data-csv`
- Optional evaluation inputs:
  - `label_col` for metric computation
  - `id_col` for stable sample identifiers in output
- Feature alignment controls:
  - `strict_features=True` fails on missing required features
  - `strict_features=False` fills missing features (strategy `zero` or `median`)

## Outputs
- Always writes:
  - `predictions.csv` via `InferenceReportWriter.write_predictions`
  - `calibration_curve.csv` (top1 compatibility output, when labels/probabilities are available)
  - `calibration_curve_<name>.csv` for each available curve (additive)
- When labels are provided:
  - metrics dict returned in-memory
  - `overall.probability_quality` namespace with calibration metrics (`ece_top1`, `ece_ovr_macro`, etc.)
  - `metrics["calibration_curves"]` map for curve payloads by curve name
  - CSV/Excel outputs under `output_dir` via `InferenceReportWriter` (e.g., `metrics.xlsx` and metrics CSV directory)
- When plots enabled:
  - ROC/PR/confusion matrix and related plots under `output_dir` (see `inference/plots.py`)
- Warnings list returned in results and printed to logs.
- Binary runs with labels present include `predicted_label` and `predicted_proba_{class}` columns
  when score outputs are probabilities.

## Internal Workflow
- Load artifacts using `ArtifactLoader`:
  - detect run type (`binary`/`meta`/`multiclass`/`hierarchical`/`legacy`)
  - load feature schema / feature list for alignment
- Load input data via `classiflow.data.load_table`.
- Align features using `FeatureAligner` and validate input via `validate_input_data`.
- Run predictions using run-type-specific predictors.
- If labels available, compute metrics and calibration quality.
- Persist outputs (CSV, Excel, plots) through report writer helpers.

## Dependencies
- Upstream callers:
  - CLI inference commands
  - Streamlit inference page (`streamlit_app/pages/06_Inference.py`)
  - Projects independent test workflow calls inference (see `projects/orchestrator.py`)
- Downstream calls:
  - bundles (`bundles/loader.py`) for bundle-based inference
  - data loading (`data/loaders.py`)
  - metrics (`inference/metrics.py`, `metrics/calibration.py`)
  - validation/drift (for warnings/reporting where integrated)
- External dependencies: `pandas`, `numpy`, `joblib`, optional `torch` depending on run type.

## Invariants & Safety Constraints
- Feature compatibility is a hard safety constraint:
  - missing training features must fail in strict mode and must be surfaced clearly in lenient mode.
- Class/probability alignment must remain consistent:
  - `meta_features.csv` and `meta_classes.csv` define ordering; inference must respect them.
  - top1 calibration uses `argmax(y_proba)` by definition; mismatch with final `y_pred` is tracked as `pred_alignment_mismatch_rate`.
- Bundle/run-dir compatibility:
  - inference must tolerate legacy manifests and produce clear errors when artifacts are incomplete.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new prediction columns (additive) | Medium | Update report writer and UI expectations; add tests for schema |
| Change feature alignment rules/defaults | High | Regression tests; document behavior; ensure strict/lenient semantics preserved |
| Change artifact discovery/loading rules | High | Bundle/run-dir roundtrip tests; update migration guidance |
| Change output filenames or workbook schemas | High | Update downstream consumers; add golden output tests |

## Testing Requirements
- Inference:
  - `pytest tests/inference/test_preprocess.py`
  - `pytest tests/inference/test_meta_predictor.py`
  - `pytest tests/inference/test_confidence.py`
- Integration:
  - `pytest tests/integration/test_meta_inference_consistency.py`
- Bundle path:
  - `pytest tests/bundles/test_bundle_roundtrip.py`

## Common Pitfalls
- Passing Parquet datasets without `pyarrow` installed (data loader will fail).
- Silent class order drift when `meta_classes.csv` is missing; loader falls back to model classes—treat this as high risk.
- Reordering columns after alignment without updating `feature_schema.json`/manifest expectations.

## Examples
```bash
classiflow infer --data test.parquet --run-dir derived --outdir inference_results --label-col diagnosis
classiflow infer --data test.parquet --bundle model.zip --fold 1 --outdir inference_results
```

## High-Risk Change Protocol
- Required design note (ADR):
  - Specify what changed in feature alignment, artifact loading, output schemas, or metrics computation.
- Required test additions:
  - Add regression tests for any previously failing bundle/run layout.
  - Add fixture-based tests for prediction column schemas and ordering.
- Required backward compatibility checks:
  - Old run directories and bundles must still load; if not, add migration tooling or clear failure messages.
- Required release note items:
  - List output schema changes, column additions, and any strict/lenient behavior changes.
