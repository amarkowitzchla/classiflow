# models

## Objective
- Define estimator registries and hyperparameter grids used by training loops.
- Provide SMOTE utilities that are safe on small minority classes.
- Provide PyTorch model wrappers used by hierarchical training and optional torch estimators.

## Public Interfaces
- Estimator registry (`src/classiflow/models/estimators.py`):
  - `get_estimators(random_state=42, max_iter=10000, logreg_params=None, resolved_device=None) -> dict[str, Any]`
  - `get_param_grids(resolved_device=None) -> dict[str, dict[str, list]]`
- SMOTE (`src/classiflow/models/smote.py`):
  - `apply_smote(X, y, k_neighbors=5, random_state=42) -> (X_res, y_res)`
  - `AdaptiveSMOTE(k_max=5, random_state=None)` with `.fit_resample(X, y)`
- Torch MLP (`src/classiflow/models/torch_mlp.py`):
  - `resolve_device(name: str) -> str`
  - `TorchMLP(nn.Module)`
  - `TorchMLPWrapper(...).fit(...).predict(...).predict_proba(...)` and `.save(...)` (used by hierarchical CV)
- Torch multiclass wrappers (`src/classiflow/models/torch_multiclass.py`):
  - `TorchLinearClassifier`, `TorchMLPClassifier` (sklearn-compatible wrappers around `backends/torch`)

## Inputs
- Estimator registries depend on:
  - `random_state` for sklearn determinism
  - `resolved_device` to decide whether torch estimators are available
- SMOTE inputs:
  - `y` should be integer-coded for `apply_smote` (hierarchical CV) and can be any array-like for `AdaptiveSMOTE` (imblearn compatibility).
- Torch wrappers require:
  - torch installation and device availability (cuda/mps/cpu)

## Outputs
- Objects returned:
  - estimator instances compatible with sklearn APIs
  - parameter grids keyed by estimator name using pipeline param prefixes (`clf__...`)
- SMOTE returns:
  - resampled arrays or original arrays if SMOTE is not applicable.

## Internal Workflow
- `AdaptiveSMOTE` adapts `k_neighbors` down to `min(minority-1, k_max)` and pass-throughs when minority size is too small.
- `get_estimators` builds a consistent set of sklearn models; optionally adds torch models when GPU-like devices are available.
- `TorchMLPWrapper` implements early stopping and batch inference; hierarchical CV calls `.save(...)` to persist `.pt` models.

## Dependencies
- Upstream callers:
  - Training: `training/nested_cv.py`, `training/multiclass.py`, `training/hierarchical_cv.py`
  - Backends: `backends/registry.py` uses torch estimators for `--backend torch`
- Downstream calls:
  - Torch multiclass wrappers delegate to `src/classiflow/backends/torch/*`.
- External dependencies: `sklearn`, `imblearn`, optional `torch`.

## Invariants & Safety Constraints
- Estimator names are part of artifact metadata and metrics tables (e.g., `model_name` fields).
- Parameter grid keys must match pipeline structure (`ImbPipeline([..., ("clf", est)])`).
- SMOTE must never crash training on small folds; “pass-through when unsafe” is a safety invariant.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new estimator option | Medium | Add tests; ensure param grid keys are correct; document expected artifacts |
| Change SMOTE adaptation rules | High | Regression tests for small-class folds; validate no new failures |
| Change torch wrapper serialization | High | Backward compatibility for saved `.pt` and joblib; update inference loaders if needed |

## Testing Requirements
- Unit: `pytest tests/unit/test_smote.py`
- Unit: `pytest tests/unit/test_torch_estimators.py`
- Training: `pytest tests/training/test_torch_backend_binary.py`
- Training: `pytest tests/training/test_torch_multiclass_estimators.py`

## Common Pitfalls
- Changing estimator defaults affects metrics and promotion gates downstream.
- Misaligned param grid keys (`clf__` vs estimator attribute names) silently disable tuning.
- Non-deterministic torch training if seeds/device settings change.
- SMOTE on very small splits can produce misleading metrics; pass-through behavior is intentional.

## Examples
```python
from classiflow.models import get_estimators, get_param_grids

estimators = get_estimators(random_state=42, max_iter=5000)
grids = get_param_grids()
print(sorted(estimators), sorted(grids))
```

