# inference/artifact_loading

## Objective
- Load inference-time artifacts from a run directory (or extracted bundle) and normalize across legacy formats.

## Public Interfaces
- `ArtifactLoader(run_dir: Path, fold: int = 1, verbose: int = 1)` in `src/classiflow/inference/loader.py`
  - `.run_type` detection: `binary`/`meta`/`multiclass`/`hierarchical`/`legacy`
  - `.load_binary_artifacts(variant="smote") -> (pipes, best_models, feature_list)`
  - `.load_meta_artifacts(variant="smote") -> (meta_model, meta_features, meta_classes, calibration_metadata)`
  - `.load_multiclass_artifacts(variant="smote") -> (...)`
  - `.get_feature_schema() -> dict` (feature_list/feature_schema handling)

## Inputs
- `run_dir` can be:
  - a base directory containing `fold{N}/...`
  - a fold directory itself (`fold{N}`), in which case `base_dir = run_dir.parent`
- Manifests:
  - prefers `base_dir/run.json` (TrainingRunManifest-style)
  - falls back to `base_dir/run_manifest.json`
- Run-type detection heuristics:
  - hierarchical: presence of `training_config.json` with `label_l2` or `hierarchical` set
  - meta: `fold_dir/binary_*/meta_model.joblib`
  - multiclass: `fold_dir/multiclass_*/multiclass_model.joblib`
  - binary: `fold_dir/binary_*/binary_pipes.joblib`

## Outputs
- Loaded objects:
  - joblib-loaded pipelines and models
  - feature/class lists from CSVs where available
  - calibration metadata dict if present

## Internal Workflow
- Manifest loading:
  - if `run.json` looks like TrainingRunManifest, map fields into `RunManifest` (inference-side type)
  - else load `RunManifest` directly from JSON
- Artifact loading functions validate expected files and raise on missing critical artifacts.
- Best-model inference for legacy formats:
  - if `binary_pipes.joblib` is not a dict payload, attempt to infer `best_models` from pipeline keys.

## Dependencies
- `joblib`, `pandas`, `json`
- Training artifact conventions from:
  - `training/meta.py` (binary/meta artifacts)
  - `training/multiclass.py` (multiclass artifacts)
  - `training/hierarchical_cv.py` (hierarchical artifacts)

## Invariants & Safety Constraints
- Artifact path conventions are part of the public API:
  - changing `binary_{variant}` or file names will break loader and bundles.
- When `meta_features.csv` is missing, loader warns that feature order may be incorrect; do not silently “guess” ordering in ways that change predictions without warnings.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new run types/artifacts (additive) | Medium | Update run-type detection and tests; document expected layout |
| Change directory layout or filenames | High | Bundle compatibility tests; migration tooling; UI adapter updates |
| Change legacy fallback behavior | High | Add tests with legacy fixtures; ensure failures are explicit and actionable |

## Testing Requirements
- `pytest tests/bundles/test_bundle_roundtrip.py`
- `pytest tests/inference/test_meta_predictor.py`

## Common Pitfalls
- Passing `--run-dir` pointing at `fold1/` vs `derived/`: loader supports both, but downstream code must treat `base_dir` correctly.
- Expecting feature lists to exist for all run types; some historical binary runs may not have them.

## Examples
```python
from classiflow.inference.loader import ArtifactLoader

loader = ArtifactLoader("derived", fold=1)
print(loader.run_type)
```

