# bundles

## Objective
- Create and load **portable model bundles** (ZIP archives) for offline inference and sharing.
- Define a stable **bundle layout** including manifests, configs, and fold artifacts.
- Provide inspection/validation utilities for bundle integrity and version compatibility.

## Public Interfaces
- Creation (`src/classiflow/bundles/create.py`):
  - `create_bundle(run_dir: Path, out_bundle: Path, fold: int|None = None, include_all_folds: bool = False, include_metrics: bool = True, description: str|None = None) -> Path`
- Inspection (`src/classiflow/bundles/inspect.py`):
  - `inspect_bundle(bundle_path: Path) -> dict`
  - `print_bundle_info(bundle_path: Path, verbose: bool = False) -> None`
  - `validate_bundle_version(bundle_path: Path, current_version: str) -> (compatible: bool, warnings: list[str])`
- Loading (`src/classiflow/bundles/loader.py`):
  - `BundleLoader(bundle_path: Path, extract_dir: Path|None = None)` (context-manager friendly)
  - `load_bundle(bundle_path: Path, fold: int = 1, extract_dir: Path|None = None) -> dict`
- CLI:
  - `classiflow bundle create|inspect|validate ...` in `src/classiflow/cli/bundle.py`

## Inputs
- `run_dir` must contain a training manifest:
  - preferred: `run.json`
  - legacy fallback: `run_manifest.json`
- Fold selection:
  - `include_all_folds=True` bundles every `fold*/` directory under `run_dir`
  - otherwise includes `fold{fold}` or defaults to fold 1
- Optional extra files included if present:
  - `training_config.json`, `inference_config.json`
  - `sanity_checks.json`, `class_order.json`, `feature_schema.json`, `final_train_config.json`, `training_stats.json`
  - metrics files matching patterns (`metrics_*.csv`, `metrics_*.xlsx`, etc.)

## Outputs
- A `.zip` file containing (minimum expected set):
  - `run.json` (training manifest)
  - `artifacts.json` (registry created during bundling)
  - `version.txt` (string like `classiflow 0.1.0`)
  - `README.txt` (bundle metadata and usage text)
  - fold directories, e.g. `fold1/...`
- `BundleLoader` extraction:
  - by default extracts to a temp directory (`classiflow_bundle_*`) and deletes it on cleanup.

## Internal Workflow
- `create_bundle`:
  - validates run directory and manifest presence
  - chooses fold directories to include
  - writes core metadata files into ZIP (`run.json`, configs, `version.txt`)
  - scans fold directories to build `artifacts.json`
  - includes selected `registry/` files and optional extras
- `BundleLoader`:
  - extracts the ZIP
  - loads `run.json` into `TrainingRunManifest` when possible
  - supports legacy conversion when bundle contains legacy manifest structure

## Dependencies
- Upstream callers:
  - CLI bundle commands
  - Inference CLI supports `--bundle` (see `cli/main.py`)
  - Project workflows may create bundles for promotion
- Downstream calls:
  - `classiflow.lineage.TrainingRunManifest`
- External dependencies: standard library (`zipfile`, `tempfile`), `pandas`/`numpy` for some metadata operations.

## Invariants & Safety Constraints
- Bundle layout is a public API:
  - required file names and paths must remain stable or be migrated explicitly.
- `version.txt` is used for compatibility warnings; format stability matters.
- `artifacts.json` is used for inspection/UI; ensure it remains valid JSON and reflects actual contents.
- Extraction must not write outside the chosen `extract_dir` (zip slip safety is delegated to `zipfile` behavior; be cautious when modifying extraction logic).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add optional files to bundle (additive) | Medium | Update docs and `README.txt` generator; add roundtrip tests |
| Change required files or paths | High | Migration plan; keep loaders tolerant; update validators and UI expectations |
| Change manifest interpretation in loader | High | Backward compatibility tests for legacy bundles and `run.json` bundles |

## Testing Requirements
- Integration: `pytest tests/bundles/test_bundle_roundtrip.py`
- Smoke: `classiflow bundle validate my_model.zip`

## Common Pitfalls
- Bundling a fold directory path instead of the base run directory (supported, but be explicit about expected layout).
- Assuming metrics files exist; bundling includes them only when present and enabled.
- Leaving temp extraction directories around (use `with BundleLoader(...)` or ensure `.cleanup()` is called).

## Examples
```bash
classiflow bundle create --run-dir derived --out model.zip --all-folds
classiflow bundle inspect model.zip --verbose
classiflow bundle validate model.zip
```

## High-Risk Change Protocol
- Required design note (ADR):
  - Describe the bundle layout change and how older bundles are handled.
  - Include an explicit “compatibility matrix” (old bundle × new code; new bundle × old code expectations).
- Required test additions:
  - Add/extend bundle roundtrip tests, including legacy manifest cases.
  - Add a fixture bundle layout test asserting required files and critical artifact paths.
- Required backward compatibility checks:
  - `classiflow infer --bundle ...` must continue to work for previously produced bundles, or fail with a clear error and remediation.
- Required release note items:
  - “Bundle format changes” with upgrade guidance and any migration tooling.

