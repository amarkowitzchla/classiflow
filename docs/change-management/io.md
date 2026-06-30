# io

## Objective
- Provide training-oriented data loading (`load_data`, `load_data_with_groups`) and schema/compatibility checks.
- Define the “classic” CSV/Parquet ingestion path used by training and inference entrypoints.
- Offer a compatibility-assessment tool to fail fast for meta/hierarchical training.

## Public Interfaces
- Loaders (`src/classiflow/io/loaders.py`):
  - `load_data(data_path, label_col, feature_cols=None, drop_na_labels=True) -> (X: pd.DataFrame, y: pd.Series)`
  - `load_data_with_groups(data_path, label_col, patient_col, feature_cols=None, ...) -> (X, y, groups)`
  - `validate_data(X: pd.DataFrame, y: pd.Series) -> None`
- Schema model (`src/classiflow/io/schema.py`):
  - `DataSchema.from_data(X, y) -> DataSchema` (pydantic model)
- Compatibility (`src/classiflow/io/compatibility.py`):
  - `CompatibilityResult` (dataclass)
  - `assess_data_compatibility(config: MetaConfig|HierarchicalConfig, return_details: bool = True) -> CompatibilityResult`
  - `print_compatibility_report(...)` (exported from `__init__`, implemented in compatibility module)

## Inputs
- `data_path` supports:
  - CSV, Parquet file, or Parquet dataset directory (delegates to `classiflow.data.load_table`)
- Required columns:
  - `label_col` must exist; `patient_col` must exist for group loading.
- `assess_data_compatibility` consumes:
  - `MetaConfig` or `HierarchicalConfig` from `classiflow.config`
  - Uses `config.resolved_data_path`

## Outputs
- `load_data*` return:
  - `X`: numeric-only features (label/patient columns excluded)
  - `y`: string labels (`astype(str)`)
  - `groups`: string group IDs
- `validate_data` raises `ValueError` for insufficient samples/features and logs warnings for NA/constant features.
- Compatibility:
  - Returns a `CompatibilityResult` including `warnings`, `errors`, and optional `DataSchema` and summaries.

## Internal Workflow
- `load_data`/`load_data_with_groups`:
  - call `classiflow.data.load_table`
  - validate required columns
  - select features (explicit or numeric auto-detect)
- `assess_data_compatibility`:
  - validate path and load table
  - run mode-specific checks for label availability, class counts, feature presence, and patient stratification needs

## Dependencies
- Upstream callers:
  - Training: `training/binary.py`, `training/meta.py`, `training/multiclass.py`
  - CLI: `cli/main.py` uses `assess_data_compatibility` before training meta/hierarchical
- Downstream calls:
  - `classiflow.data.load_table`
  - `classiflow.io.schema.DataSchema`
- External dependencies: `pandas`, `numpy`, `pydantic`, optional `pyarrow` (via `data`).

## Invariants & Safety Constraints
- Auto feature selection must exclude label/patient columns.
- Explicit `feature_cols` must not include label or patient columns (fail closed).
- Label casting to string is part of the data contract; downstream tasks assume string class names.
- Compatibility checks must fail closed (incompatible ⇒ `is_compatible=False`) and provide actionable errors.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Improve error messages / suggestions | Low | Update unit tests as needed |
| Change `validate_data` thresholds (min samples/features) | Medium | Update tests and CLI docs |
| Change compatibility rules for meta/hierarchical | High | Add regression tests; update CLI gating behavior |

## Testing Requirements
- Unit: `pytest tests/unit/test_compatibility.py`
- Unit: `pytest tests/data/test_loaders.py`

## Common Pitfalls
- Confusing `classiflow.data.load_data(spec)` vs `classiflow.io.load_data(path, ...)` (different API shapes).
- Treating `groups` as numeric; this module standardizes groups to strings.
- Forgetting to pass `feature_cols` leads to numeric auto-detection that may differ across datasets.

## Examples
```python
from pathlib import Path
from classiflow.io import load_data_with_groups

X, y, groups = load_data_with_groups(Path("data.parquet"), label_col="diagnosis", patient_col="patient_id")
```
