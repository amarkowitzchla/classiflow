# inference/predictors

## Objective
- Convert aligned feature tables into predictions for each supported run type with stable output column schemas.

## Public Interfaces
- Predictors in `src/classiflow/inference/predict.py`:
  - `BinaryPredictor(pipes: dict, best_models: dict, task_names: list[str]|None = None).predict(X: pd.DataFrame) -> pd.DataFrame`
  - `MetaPredictor(meta_model, meta_features: list[str], meta_classes: list[str], calibration_metadata: dict|None = None).predict(binary_predictions: pd.DataFrame) -> pd.DataFrame`
  - `HierarchicalPredictor(models: dict, config: dict, device="cpu").predict(X: pd.DataFrame) -> pd.DataFrame`
  - `MulticlassPredictor(...)` (used when run type is multiclass)

## Inputs
- `BinaryPredictor`:
  - expects pipeline keys of form `{task}__{model_name}` and `best_models[task] = model_name`
  - outputs rely on per-task scoring via `predict_proba` or `decision_function`
- `MetaPredictor`:
  - requires `binary_predictions` to contain all columns listed in `meta_features` (order matters)
  - uses `meta_classes` when present to name per-class probability columns
- Hierarchical predictor:
  - expects a dict including `l1_model`, `scaler`, encoders, and optional branch models/encoders (artifact loader supplies)

## Outputs
- `BinaryPredictor.predict` columns (per task):
  - `{task}_score`
  - `{task}_pred`
- `MetaPredictor.predict` columns:
  - `predicted_label` / `y_pred`
  - `predicted_proba_{class}` and `y_prob_{class}` for each class (when proba is available)
  - `predicted_proba` / `y_prob` (max probability)
  - `y_score_raw` and `y_score_raw_{class}` (best-effort from uncalibrated model)
  - calibration metadata columns (`calibration_method`, `calibration_enabled`, etc.)

## Internal Workflow
- Score extraction:
  - `BinaryPredictor` thresholds scores at 0.5 for probas, else 0.0 for decision values.
- Feature alignment:
  - predictors call `_align_features_for_model(...)` to match trained feature names to provided DataFrames.
- `MetaPredictor` fills NaNs with 0.0 after aligning and ordering meta features.

## Dependencies
- Artifact conventions and saved metadata:
  - `meta_features.csv`, `meta_classes.csv`, `best_models` mapping
- External dependencies: `pandas`, `numpy`.

## Invariants & Safety Constraints
- Column naming and ordering are public API:
  - downstream report writers and project workflows expect stable names (`y_pred`, `y_prob_*` patterns).
- Any change that affects feature alignment or task naming can silently change predictions; changes must be treated as high risk.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new prediction columns (additive) | Medium | Update docs and tests; ensure report writer tolerates additions |
| Rename existing columns | High | Migration plan; update report writers, UI, and project parsers |
| Change thresholding behavior | High | Regression tests; document behavioral change and expected impacts |

## Testing Requirements
- `pytest tests/inference/test_meta_predictor.py`
- `pytest tests/integration/test_meta_inference_consistency.py`

## Common Pitfalls
- Missing meta-feature columns leads to hard failure; ensure `meta_features.csv` is present in artifacts.
- Model proba outputs must be aligned to class names; mismatches produce misleading per-class columns.

## Examples
```python
from classiflow.inference.predict import BinaryPredictor

pred = BinaryPredictor(pipes, best_models).predict(X_df)
print(pred.filter(like="_score").head())
```

