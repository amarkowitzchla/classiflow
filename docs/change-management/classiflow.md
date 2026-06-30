# classiflow

## Objective
- Define the package’s high-level identity and version (`__version__`).
- Provide a **stable, minimal** import surface for common entrypoints.
- Re-export training/task/config primitives used by notebooks and downstream code.
- Act as a convenient “front door” without forcing heavy imports at import time.

## Public Interfaces
- `__version__ = "0.1.0"`
- Re-exports from `src/classiflow/__init__.py`:
  - `train_binary_task(config: TrainConfig) -> dict`
  - `train_meta_classifier(config: MetaConfig) -> dict`
  - `TaskBuilder(classes: list[str])`
  - `TrainConfig`, `MetaConfig`

## Inputs
- Import-time dependencies:
  - Imports `classiflow.training.binary.train_binary_task`
  - Imports `classiflow.training.meta.train_meta_classifier`
  - Imports `classiflow.tasks.builder.TaskBuilder`
  - Imports `classiflow.config.TrainConfig`, `classiflow.config.MetaConfig`

## Outputs
- Python module exports via `__all__`.
- No filesystem side effects.

## Internal Workflow
- Set module docstring + `__version__`.
- Import and expose selected callables/types in `__all__`.

## Dependencies
- Upstream callers: external users importing `classiflow` directly; CLI (`classiflow.cli.main`) imports `__version__`.
- Downstream calls: none (this module is mostly re-exports).
- External dependencies: none directly (but re-exported modules depend on pandas/sklearn/etc).

## Invariants & Safety Constraints
- `__version__` is treated as a compatibility indicator for bundles/manifests (`bundles/create.py`, `lineage/manifest.py`).
- Re-exports should remain **stable** or follow explicit deprecation (avoid silent rename/removal).
- Avoid adding heavy imports that slow `import classiflow` or pull in optional deps unexpectedly.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new re-export | Low | Add unit coverage for import; ensure no heavy import-time side effects |
| Remove/rename a re-export | Medium | Deprecation window; update docs; update downstream imports/tests |
| Change `__version__` scheme/semantics | High | Ensure bundle/version checks still behave; update release notes and compatibility policy |

## Testing Requirements
- Unit: `pytest tests/unit/test_backend_registry.py` (covers some import wiring patterns)
- Smoke: `python -c "import classiflow; print(classiflow.__version__)"`

## Common Pitfalls
- Adding optional-dependency imports here (e.g., torch/pyarrow) can break base installs.
- Re-exporting symbols that are not part of the intended public API increases change burden.
- Circular imports: keep this module “thin”.

## Examples
```python
import classiflow
from classiflow import TrainConfig, train_binary_task

print(classiflow.__version__)
```

