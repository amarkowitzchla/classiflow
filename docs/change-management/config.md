# config

## Objective
- Centralize **training and inference configuration** as dataclasses.
- Provide a single place for **path resolution** and backward-compatible flags (`--data` vs legacy `--data-csv`).
- Define the “contract” for parameters that affect splitting, SMOTE behavior, backend selection, and reproducibility.

## Public Interfaces
- `_resolve_data_path(data: Path|None, data_csv: Path|None) -> Path`
- Dataclasses in `src/classiflow/config.py`:
  - `TrainConfig`
  - `MetaConfig(TrainConfig)`
  - `MulticlassConfig(TrainConfig)`
  - `HierarchicalConfig`
- Methods/properties used across the codebase:
  - `TrainConfig.resolved_data_path -> Path`
  - `TrainConfig.to_dict() -> dict`
  - `TrainConfig.save(path: Path) -> None`
  - `MetaConfig.to_dict() -> dict`
  - `HierarchicalConfig.resolved_data_path -> Path`
  - `HierarchicalConfig.to_dict() -> dict`
  - `HierarchicalConfig.save(path: Path) -> None`

## Inputs
- CLI flags map 1:1 into these configs (see `src/classiflow/cli/main.py` and `src/classiflow/cli/stats.py`).
- Common fields:
  - Data: `data_path` (preferred), `data_csv` (deprecated), `label_col`, `feature_cols`, `patient_col`
  - CV: `outer_folds`, `inner_splits`, `inner_repeats`, `random_state`
  - SMOTE: `smote_mode`, `smote_k_neighbors`
  - Backend/device: `backend`, `device`, `model_set`, `torch_*`, `require_torch_device`
- Hierarchical-specific fields:
  - `label_l1`, `label_l2`, `l2_classes`, `min_l2_classes_per_branch`, `output_format`
  - Torch MLP hyperparameters: `mlp_epochs`, `mlp_batch_size`, `mlp_hidden`, `mlp_dropout`, `early_stopping_patience`

## Outputs
- JSON config persistence:
  - `TrainConfig.save()` writes JSON (paths stringified).
  - `HierarchicalConfig.save()` writes JSON (paths stringified).
- Runtime behavior:
  - Emits a warning when legacy `data_csv` is used via `_resolve_data_path()`.

## Internal Workflow
- Normalize `Path`-like fields in `__post_init__`.
- Cross-set `data_path` and `data_csv` for forward/backward compatibility.
- Defer “must be provided” validation to callers via `resolved_data_path`.
- Serialize with `dataclasses.asdict` and convert `Path` fields to strings.

## Dependencies
- Upstream callers:
  - CLI: `src/classiflow/cli/main.py`, `src/classiflow/cli/stats.py`
  - Training: `src/classiflow/training/*.py`
  - Compatibility checks: `src/classiflow/io/compatibility.py`
  - Inference: `src/classiflow/inference/config.py` (separate, but follows same pattern)
- Downstream calls: none (this module is pure configuration).
- External dependencies: standard library only.

## Invariants & Safety Constraints
- `random_state` must propagate to:
  - split generators (`splitting/`)
  - sklearn estimators / grid search
  - torch seeding (when torch backend is used)
- The `data_path` vs `data_csv` compatibility behavior must not silently change; CLI examples and migration tooling depend on it.
- Patient-level stratification configuration (`patient_col`) must remain explicit and stable; it is a safety control for leakage prevention.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new config field with safe default | Low | Add unit coverage; ensure `to_dict()` includes it if needed |
| Change defaults for CV / SMOTE / backend | High | Add regression tests; update docs and CLI help; ensure determinism |
| Remove/rename a field used by CLI or project YAML | High | Deprecation + migration plan; update docs, tests, and project scaffolds |
| Change serialization (`to_dict()`/`save`) format | High | Backward compatibility tests for `run.json`/`training_config.json` consumers |

## Testing Requirements
- Unit: `pytest tests/unit/test_compatibility.py`
- Unit: `pytest tests/training/test_patient_stratified_wiring.py`
- Smoke: `python -c "from classiflow.config import TrainConfig; print(TrainConfig().to_dict())"`

## Common Pitfalls
- Treating `data_csv` as equivalent to `data_path` in new code without preserving deprecation warning.
- Forgetting to convert new `Path` fields to strings in `to_dict()` (breaks JSON serialization).
- Introducing config changes that subtly alter fold counts/seed semantics (causes non-obvious metric drift).
- Conflating `patient_col` and `group_col` (there is legacy aliasing; keep behavior explicit).

## Examples
```python
from classiflow.config import TrainConfig

cfg = TrainConfig(data_path="data.parquet", label_col="diagnosis", patient_col="patient_id")
print(cfg.resolved_data_path)
```

