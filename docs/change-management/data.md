# data

## Objective
- Provide a unified, format-aware data loading layer (CSV / Parquet file / Parquet dataset directory).
- Encode the **data contract** for turning raw tabular input into ML-ready arrays and metadata.
- Centralize column validation and feature selection to reduce schema drift across pipelines.

## Public Interfaces
- Types:
  - `DataFormat` (`csv`, `parquet`, `parquet_dataset`) in `src/classiflow/data/spec.py`
  - `DataSpec` in `src/classiflow/data/spec.py`
  - `LoadedDataset` in `src/classiflow/data/dataset.py`
- Loaders (`src/classiflow/data/loaders.py`):
  - `validate_parquet_available() -> None`
  - `infer_format(path: Path) -> DataFormat`
  - `load_table(path: Path, columns: list[str]|None = None, filters: dict|None = None) -> pd.DataFrame`
  - `load_data(spec: DataSpec) -> LoadedDataset`
- Validation (`src/classiflow/data/validation.py`):
  - `validate_columns(...) -> list[str]` (raises on missing)
  - `validate_features(X: pd.DataFrame, strict: bool = True) -> list[str]`
  - `generate_missingness_report(df: pd.DataFrame, feature_cols: list[str]|None = None) -> dict`

## Inputs
- Supported input sources:
  - `.csv` file
  - `.parquet` file
  - directory containing `**/*.parquet` (dataset)
- `DataSpec` fields controlling behavior:
  - `label_col`, `id_col`, `group_col`
  - `feature_cols` (explicit) vs auto-detected numeric columns (sorted)
  - `classes` (filter/order)
  - `columns` (subset read optimization)
  - `dataset_glob`, `filters` (dataset directory support)

## Outputs
- `load_table` returns raw `pd.DataFrame` without ML shaping.
- `load_data` returns `LoadedDataset`:
  - `X: np.ndarray` (float32, deterministic feature order)
  - `y: np.ndarray|None` (string labels if `label_col` provided)
  - `feature_names: list[str]`
  - `ids`, `groups`, and `df_meta` (optional)

## Internal Workflow
- Infer format via `DataFormat.from_path`.
- Load raw data using the appropriate backend:
  - CSV: `pd.read_csv`
  - Parquet: `pyarrow.parquet.read_table(...).to_pandas()`
  - Parquet dataset: `pyarrow.dataset.dataset(...).to_table(...).to_pandas()`
- Validate required columns with `validate_columns`.
- Filter rows with missing labels (training use).
- Filter to selected classes if configured.
- Determine feature columns:
  - explicit `feature_cols`, else numeric-only auto-detection with a stable sort.
- Validate feature quality (warnings by default in `load_data`).

## Dependencies
- Upstream callers:
  - CLI: `src/classiflow/cli/main.py` uses `load_table` for hierarchical inference/training.
  - Training/inference: `io/loaders.py` delegates to `classiflow.data.load_table`.
  - Compatibility checks: `io/compatibility.py` uses `load_table`.
- Downstream calls:
  - `classiflow.data.validation.validate_columns`, `validate_features`
- External dependencies: `pandas`, `numpy`, optional `pyarrow` for Parquet.

## Invariants & Safety Constraints
- Feature selection must be deterministic:
  - when auto-detecting features, order is `sorted(numeric_columns)`.
- `LoadedDataset.X` must be float32 for consistent downstream estimator behavior/memory.
- When `classes` is provided, ordering is meaningful and must be preserved.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add optional DataSpec fields | Low | Unit tests for parsing/serialization |
| Change auto feature selection rules | High | Regression tests; update manifests/inference aligner expectations |
| Change Parquet dataset loading behavior | Medium | Add tests covering dataset directories; document pyarrow requirement |

## Testing Requirements
- Unit: `pytest tests/data/test_loaders.py`
- Smoke: `python -c "from classiflow.data import DataSpec, load_data; ..."` (use a small local fixture)

## Common Pitfalls
- Assuming label columns are numeric; `load_data` casts labels to `str`.
- Feature leakage via accidental inclusion of label/id/group columns in `feature_cols`.
- Missing `pyarrow` when using Parquet: call `validate_parquet_available()` early if needed.
- Dataset directory loads can be large; use `columns` to constrain reads.

## Examples
```python
from pathlib import Path
from classiflow.data import DataSpec, load_data

spec = DataSpec(path=Path("data.parquet"), label_col="diagnosis", group_col="patient_id")
dataset = load_data(spec)
print(dataset.X.shape, len(dataset.feature_names))
```

