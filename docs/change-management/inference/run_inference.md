# inference/run_inference

## Objective
- Implement the orchestrated inference pipeline that loads artifacts, aligns data, predicts, evaluates, and writes reports.

## Public Interfaces
- `run_inference(config: InferenceConfig) -> dict[str, Any]` in `src/classiflow/inference/api.py`

## Inputs
- `InferenceConfig` fields:
  - `run_dir: Path` (may be base run dir or `fold{N}` dir)
  - `data_path`/`data_csv` (CSV/Parquet/dataset directory)
  - `output_dir: Path`
  - optional `id_col`, `label_col`
  - `strict_features`, `lenient_fill_strategy`
  - `max_roc_curves`, `include_plots`, `include_excel`
  - `device`, `batch_size`, `verbose`

## Outputs
- Returns `results` dict including:
  - `predictions: pd.DataFrame`
  - optional `metrics: dict`
  - `output_files: dict[str, Path]`
  - `warnings: list[str]`
- Writes at least:
  - `output_dir/predictions.csv`
- Binary runs with labels provided emit `predicted_label` and `predicted_proba_{class}` columns
  when score outputs are probabilities.
- Writes when labels exist and Excel enabled:
  - `output_dir/metrics.xlsx` plus metrics CSV files under writer-managed subdirectories (see `inference/reports.py`)

## Internal Workflow
- `[1/7]` load artifacts via `ArtifactLoader`
- `[2/7]` load data via `classiflow.data.load_table`, validate with `validate_input_data`
- align features via `FeatureAligner.align(...)` producing:
  - `X` (aligned features DataFrame)
  - `metadata` (id/label columns subset)
  - alignment warnings (missing/extra features, fill behavior)
- `[3/7]` predict via `_run_predictions(...)` (routes based on run type)
- `[4/7]` compute metrics (only if `label_col` exists in metadata)
- `[5/7]` generate plots (gated by `include_plots` and label availability)
- `[6/7]` write reports (predictions CSV always; workbook optional)

## Dependencies
- `ArtifactLoader` (`inference/loader.py`)
- `FeatureAligner` + `validate_input_data` (`inference/preprocess.py`)
- predictors (`inference/predict.py`)
- metrics (`inference/metrics.py`, `metrics/calibration.py`)
- outputs (`inference/reports.py`, `inference/plots.py`)

## Invariants & Safety Constraints
- The returned `predictions` DataFrame must include:
  - `y_true` (if labels provided else NaN)
  - `sample_id` (from `id_col` or index)
  - `split` and `fold_id` metadata columns
- Strict vs lenient feature handling must not change silently; it affects model validity.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new output files (additive) | Medium | Update docs and UI artifact allowlists if needed |
| Change prediction column naming | High | Regression tests; update report writer and consumers |
| Change label/ID handling logic | High | Update tests; ensure downstream project workflows still parse outputs |

## Testing Requirements
- `pytest tests/inference/test_preprocess.py`
- `pytest tests/integration/test_meta_inference_consistency.py`

## Common Pitfalls
- Using `config.label_col` that is not present in input data: metrics will be skipped, which may surprise users.
- Missing training feature list in artifacts triggers fallback to numeric columns; treat this as compatibility risk.

## Examples
```python
from classiflow.inference import InferenceConfig, run_inference

cfg = InferenceConfig(run_dir="derived", data_path="test.parquet", output_dir="inference_results", label_col="diagnosis")
results = run_inference(cfg)
```
