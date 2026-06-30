# lineage

## Objective
- Provide **run manifests** and hashing utilities for reproducibility and provenance.
- Give trained artifacts a stable identity (`run_id`) and bind them to input data hashes.
- Support compatibility checks between training features and inference features.

## Public Interfaces
- Hashing (`src/classiflow/lineage/hashing.py`):
  - `compute_file_hash(file_path: Path, algorithm="sha256", chunk_size=8192) -> str`
  - `compute_dataframe_hash(df: pd.DataFrame, algorithm="sha256", canonical=True) -> str`
  - `compute_canonical_hash(data, algorithm="sha256") -> str`
  - `get_file_metadata(file_path: Path) -> dict` (supports Parquet dataset directories)
- Manifests (`src/classiflow/lineage/manifest.py`):
  - `TrainingRunManifest` (dataclass) with `.save()` / `.load()`
  - `InferenceRunManifest` (dataclass) with `.save()` / `.load()`
  - `create_training_manifest(...) -> TrainingRunManifest`
  - `create_inference_manifest(...) -> InferenceRunManifest`
  - `load_training_manifest(run_dir: Path) -> TrainingRunManifest`
  - `validate_manifest_compatibility(training_manifest, inference_features) -> (compatible: bool, warnings: list[str])`

## Inputs
- Training:
  - data path (file or parquet dataset directory)
  - data hash (typically `get_file_metadata(... )["sha256_hash"]`)
  - `config` dict (often `TrainConfig.to_dict()` + extra)
  - `feature_list` (explicitly recorded)
  - `task_definitions` (e.g., `"binary_task": "positive_class=..."`)
- Inference:
  - `parent_run_id` and inference data metadata
  - model source (bundle path or run dir) optionally recorded

## Outputs
- JSON manifests:
  - training: `run.json` (preferred) or legacy `run_manifest.json`
  - inference: `inference_run.json`-like outputs are created by inference flows (see `inference/enhanced_api.py`)
- Compatibility warnings:
  - missing features, extra features, and order differences

## Internal Workflow
- Manifests are dataclasses serialized with `json.dump(..., default=str)` to handle `Path` objects inside payloads.
- `create_training_manifest` attempts to capture:
  - python version, hostname
  - git hash via `git rev-parse HEAD` (best-effort)
- `get_file_metadata`:
  - handles parquet dataset directories by hashing file list + per-file hashes to produce a stable directory hash.

## Dependencies
- Upstream callers:
  - Training: `training/binary.py`, `training/multiclass.py`, `training/meta.py` create and save `run.json`.
  - Bundles: `bundles/create.py` includes `run.json` and `version.txt`.
  - UI: `ui_api` parses manifests to display run metadata.
- Downstream calls:
  - `hashing.compute_file_hash` used across code.
- External dependencies: `pandas`, `numpy`, optional `pyarrow` for parquet metadata-based row counts.

## Invariants & Safety Constraints
- `run.json` schema is part of the public API:
  - keys like `run_id`, `training_data_hash`, `feature_list`, `task_definitions` are consumed by inference/bundles/UI.
- Hash computation must be stable and deterministic for a given dataset state.
- `validate_manifest_compatibility` must conservatively report missing features as incompatibility.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add optional manifest fields | Medium | Ensure JSON remains readable by old code; update adapters if needed |
| Rename/remove manifest keys | High | Migration tooling; adapter updates; regression tests for old manifests |
| Change directory hash algorithm | High | Requires explicit versioning; impacts run identity and reproducibility |

## Testing Requirements
- Unit: `pytest tests/lineage/test_hashing.py`
- Unit: `pytest tests/lineage/test_manifest.py`
- Integration: `pytest tests/bundles/test_bundle_roundtrip.py`

## Common Pitfalls
- Writing non-JSON-serializable objects into `config` payloads without `default=str` coverage.
- Treating feature order differences as harmless: inference aligns/reorders features, but missing features are fatal.
- Assuming `pyarrow` is installed when hashing/parquet metadata is requested.

## Examples
```python
from pathlib import Path
from classiflow.lineage.hashing import get_file_metadata
from classiflow.lineage.manifest import create_training_manifest

md = get_file_metadata(Path("data.parquet"))
manifest = create_training_manifest(
    data_path=Path("data.parquet"),
    data_hash=md["sha256_hash"],
    data_size_bytes=md["size_bytes"],
    data_row_count=md.get("row_count"),
    config={"label_col": "diagnosis"},
    task_type="binary",
    feature_list=["f1", "f2"],
)
manifest.save(Path("derived/run.json"))
```

