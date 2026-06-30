# backends

## Objective
- Provide a **backend registry** to select estimator families and tuning grids based on `--backend` and `--model-set`.
- Encapsulate torch-backed estimators behind sklearn-compatible interfaces for use in grid search/pipelines.

## Public Interfaces
- `get_backend(name: str|None) -> str` in `src/classiflow/backends/registry.py`
- `get_model_set(command: str, backend: str, model_set: str|None, *, ...) -> dict` in `src/classiflow/backends/registry.py`
- Torch estimator wrappers (used via the registry):
  - `TorchLogisticRegressionClassifier`, `TorchMLPClassifier`, `TorchSoftmaxRegressionClassifier`, `TorchMLPMulticlassClassifier` in `src/classiflow/backends/torch/estimators.py`
- Torch utilities:
  - `resolve_device`, `resolve_dtype`, `set_seed`, `make_dataloader` in `src/classiflow/backends/torch/utils.py`

## Inputs
- Backend selection:
  - `backend` must be `"sklearn"` or `"torch"` (normalized by `get_backend`).
- Model-set selection:
  - torch: `"torch_basic"` or `"torch_fast"`
  - sklearn: `"default"` only
- `get_model_set` behavior depends on `command`:
  - `"train-binary"` returns `{"estimators", "param_grids"}`
  - `"train-meta"` returns base + meta estimator sets:
    - `{"base_estimators","base_param_grids","meta_estimators","meta_param_grids"}`

## Outputs
- Dicts of estimators and parameter grids compatible with:
  - `imblearn.pipeline.Pipeline([("sampler", ...), ("scaler", ...), ("clf", est)])`
  - `sklearn.model_selection.GridSearchCV`
- For torch backend, estimators are sklearn-like objects that implement `fit`, `predict`, and `predict_proba` and are joblib-serializable via `__getstate__/__setstate__`.

## Internal Workflow
- The registry hard-codes:
  - default estimator constructors and default grids per `model_set`
  - training hyperparameters (epochs, patience, dropout) for torch estimators
- Torch utilities enforce:
  - device fallback behavior (`auto`, `cuda`, `mps`, `cpu`)
  - dtype guardrails (float16 disallowed on cpu/mps)
  - deterministic seeding (`cudnn.deterministic=True`, `benchmark=False`)

## Dependencies
- Upstream callers:
  - Training: `training/binary.py` and `training/meta.py` call `get_model_set(...)`
  - Tests: backend registry and torch estimator tests
- Downstream calls:
  - `classiflow.models.estimators` for sklearn defaults
- External dependencies: `sklearn`, `torch` (for torch backend), `numpy`.

## Invariants & Safety Constraints
- Backend/model-set selection is user-facing API (CLI flags and project YAML hints); do not change accepted strings silently.
- Parameter grid keys must match pipeline naming (`clf__...`) and estimator parameter names.
- Torch device fallback must be explicit and logged; `require_torch_device=True` must fail fast when requested device is unavailable.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new torch model_set (additive) | Medium | Add tests; document tradeoffs; ensure deterministic defaults |
| Change torch estimator defaults/grids | High | Regression tests for metrics drift; update docs; consider promotion thresholds impact |
| Change serialization semantics of torch estimators | High | Bundle/inference roundtrip tests; backward compatibility guidance |

## Testing Requirements
- Unit: `pytest tests/unit/test_backend_registry.py`
- Unit: `pytest tests/unit/test_torch_estimators.py`
- Training: `pytest tests/training/test_torch_backend_binary.py`

## Common Pitfalls
- GridSearchCV + torch training can be slow; `torch_fast` exists to bound runtime.
- Device resolution differences between `backends/torch/utils.py` and `models/torch_mlp.py`—keep semantics aligned.
- Joblib serialization depends on `_input_dim` and `_num_classes` being set; changes can break load-from-bundle flows.

## Examples
```python
from classiflow.backends import get_model_set

spec = get_model_set(command="train-binary", backend="torch", model_set="torch_fast", device="auto")
print(spec["estimators"].keys())
```

