# artifacts

## Objective
- Provide thin utilities for persisting and loading **training artifacts** (models, metrics tables).
- Standardize a small set of filenames used across training and inference flows.
- Offer backward compatibility for legacy model serialization formats.

## Public Interfaces
- `save_nested_cv_results(results: dict, outdir: Path) -> None` (`classiflow.artifacts.saver`)
- `save_model(model: Any, path: Path, metadata: dict|None = None) -> None` (`classiflow.artifacts.saver`)
- `load_model(path: Path) -> tuple[Any, dict]` (`classiflow.artifacts.loader`)
- `load_meta_pipeline(fold_dir: Path, variant: str = "smote") -> dict` (`classiflow.artifacts.loader`)

## Inputs
- `save_nested_cv_results` expects `results` shaped like `NestedCVOrchestrator.run_single_task()`:
  - keys: `inner_cv_rows`, `inner_cv_split_rows`, `outer_rows`
- `load_model` reads `.joblib` created by `save_model` (payload dict with `model` and optional `metadata`), but also supports legacy “model-only” joblib.
- `load_meta_pipeline` expects fold layout:
  - `fold{N}/binary_{variant}/binary_pipes.joblib`
  - `fold{N}/binary_{variant}/meta_model.joblib`
  - `fold{N}/binary_{variant}/meta_features.csv`
  - `fold{N}/binary_{variant}/meta_classes.csv`

## Outputs
- Metrics tables:
  - `metrics_inner_cv.csv`
  - `metrics_inner_cv_splits.csv`
  - `metrics_inner_cv_splits.xlsx` (best-effort)
  - `metrics_outer_binary_eval.csv`
- Serialized model payloads:
  - `.joblib` files created by `joblib.dump`

## Internal Workflow
- Use pandas to materialize result rows into CSV/Excel.
- Use `SCORER_ORDER` from `classiflow.metrics.scorers` to define stable column ordering for split metrics.
- Store/restore joblib payloads with a compatibility branch for legacy formats.

## Dependencies
- Upstream callers:
  - Training: `src/classiflow/training/binary.py` uses `save_nested_cv_results`
  - Inference helpers / project orchestration may use `load_meta_pipeline` patterns
- Downstream calls:
  - `classiflow.metrics.scorers.SCORER_ORDER`
- External dependencies: `pandas`, `joblib`, optional `xlsxwriter` for Excel.

## Invariants & Safety Constraints
- Filenames and CSV schemas are effectively public API for downstream scripts and the Streamlit UI (`streamlit_app/ui/helpers.py`).
- `metrics_inner_cv_splits.csv` column order must remain stable: `["task_model","outer_fold","inner_split"] + SCORER_ORDER`.
- `load_model` must continue accepting legacy “model-only” joblib payloads.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Refactor internal save/load helpers | Low | Unit tests for serialization roundtrip |
| Change filenames or CSV column schemas | High | Add migration notes; update consumers; add golden tests |
| Change joblib payload format | High | Backward compat loader; add regression tests for old format |

## Testing Requirements
- Unit: `pytest tests/unit/test_metrics.py` (metrics table expectations are often coupled)
- Bundle/inference smoke: `pytest tests/bundles/test_bundle_roundtrip.py`

## Common Pitfalls
- Writing Excel output unconditionally (xlsxwriter may be missing).
- Accidentally changing CSV column order (breaks downstream parsing and comparisons).
- Loading artifacts from an unexpected directory layout (fold vs base run dir).

## Examples
```python
from pathlib import Path
from classiflow.artifacts import load_model

model, metadata = load_model(Path("fold1/binary_smote/meta_model.joblib"))
```

