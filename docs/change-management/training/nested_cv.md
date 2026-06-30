# training/nested_cv

## Objective
- Implement nested cross-validation orchestration for **binary** classification tasks.
- Provide a reusable inner-CV + outer-eval loop with optional patient-level splits.

## Public Interfaces
- `NestedCVOrchestrator(...)` in `src/classiflow/training/nested_cv.py`
  - `run_single_task(X: pd.DataFrame, y: pd.Series, task_name="binary_task", outdir: Path|None = None, groups: np.ndarray|None = None, patient_col: str|None = None) -> dict`

## Inputs
- `X`: pandas DataFrame of numeric features.
- `y`: pandas Series of binary labels (0/1 expected in practice).
- `groups` + `patient_col`:
  - when provided, uses `splitting.iter_outer_splits` / `iter_inner_splits` and enforces leakage checks.
- `smote_mode`:
  - `"off"` → only `"none"` variant
  - `"on"` or `"both"` → runs `"smote"` and `"none"` variants (note: current implementation returns `["smote","none"]` when `"on"` or `"both"`)

## Outputs
- Returns a dict containing:
  - `inner_cv_rows`, `inner_cv_split_rows`, `outer_rows`, and per-fold metadata.
- Writes plots when `outdir` is provided:
  - `fold{N}/roc_{task}_fold{N}.png`, `pr_{task}_fold{N}.png`, `cm_{task}_fold{N}.png`
  - `roc_{task}_averaged.png`, `pr_{task}_averaged.png`
- Does **not** persist trained pipelines/models to disk; consumers should not assume inference-ready artifacts exist for this mode.

## Internal Workflow
- Outer CV:
  - group-aware if `groups` provided, else `StratifiedKFold`
- Inner CV:
  - group-aware with `iter_inner_splits` when patient grouping is enabled
  - adapts effective `n_splits` down based on minority class size to avoid failures
- For each estimator:
  - build `ImbPipeline([("sampler", ...), ("scaler", StandardScaler()), ("clf", est)])`
  - `GridSearchCV(..., scoring=get_scorers(), refit="F1 Score")`
  - evaluate best estimator on train and outer-val via `compute_binary_metrics`

## Dependencies
- `classiflow.models.AdaptiveSMOTE`
- `classiflow.metrics.scorers.get_scorers` and `SCORER_ORDER`
- `classiflow.metrics.binary.compute_binary_metrics`
- `classiflow.splitting.*` for patient-safe splitting
- `classiflow.plots.*` for fold plots

## Invariants & Safety Constraints
- Patient leakage checks must run when group splitting is enabled.
- Inner CV split adaptation must preserve safety (avoid invalid CV) without silently changing the meaning of folds.
- Refitting metric (`"F1 Score"`) is part of behavioral API; changing it changes model selection.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Refactor without changing CV semantics | Medium | Regression tests for outputs; ensure no leakage checks removed |
| Change refit metric / scoring set | High | Update docs/tests; expect metrics drift; align downstream promotion gates |
| Persist models to disk (new behavior) | High | Define artifact schema; update inference/bundles expectations and tests |

## Testing Requirements
- Indirectly covered via training wiring and meta training:
  - `pytest tests/training/test_patient_stratified_wiring.py`
  - `pytest tests/unit/test_metrics.py`

## Common Pitfalls
- Assuming `groups` aligns to `X.index` (it must).
- Passing non-binary `y` leads to misleading metrics; callers should enforce binary conversion.
- Confusing “sampler=smote” with “smote_mode=on/both”; verify intended variant behavior before changing.

## Examples
```python
from classiflow.training.nested_cv import NestedCVOrchestrator

orc = NestedCVOrchestrator(outer_folds=3, inner_splits=5, inner_repeats=2, random_state=42, smote_mode="off")
results = orc.run_single_task(X, y, outdir=None)
```

