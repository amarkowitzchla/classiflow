# ADR 0007: Mode/Engine-Aware Project Configuration Schema

## Status
Accepted

## Context
`classiflow project bootstrap` generated `project.yaml` files with mixed backend/runtime settings (`backend`,
`device`, `torch_*`) regardless of task mode. This created ambiguous UX, especially for multiclass workflows where
Torch estimators were configured under `backend: sklearn`.

Affected high-risk modules: `projects`, `training`, `cli`.
Relevant contracts:
- `docs/change-management/projects.md`
- `docs/change-management/cli.md`
- `docs/change-management/training/multiclass.md`

## Decision
1. Replace top-level backend/runtime keys with an `execution` block:
   - `execution.engine`: `sklearn|torch|hybrid`
   - `execution.device`: required for `torch|hybrid`
   - `execution.torch`: required for `torch|hybrid`
2. Introduce mode/engine conditional validation in `ProjectConfig`.
3. Make multiclass runtime explicit via `multiclass.backend`:
   - `sklearn_cpu`
   - `torch_{auto|cpu|cuda|mps}`
   - `hybrid_sklearn_meta_torch_base`
4. Add config discoverability commands:
   - `classiflow config show|explain|validate|normalize`
5. Update bootstrap UX:
   - new flags `--engine`, `--device`, `--show-options`
   - emit minimal mode/engine-aware YAML with only relevant sections.
6. Keep one-release backward compatibility by normalizing legacy keys during load/validation.

## Consequences
### Positive
- Unambiguous backend/runtime selection.
- Smaller, mode-aware generated YAML.
- Users can discover allowed values and validate configs without source inspection.
- Clear error paths and suggestions for invalid combinations.

### Negative / tradeoffs
- New schema introduces stricter validation and may reject previously tolerated combinations.
- `classiflow config` adds additional CLI surface to maintain.

## Compatibility
- CLI flags:
  - Added `project bootstrap --engine --device --show-options`.
  - Added new `config` command group.
- Artifacts/manifests/bundles:
  - No change to run artifact filenames/layout.
- Migrations/deprecations:
  - Legacy keys (`backend`, `device`, `torch_*`, `model_set`, `multiclass.estimator_mode`) are normalized to the new schema.
  - `classiflow config normalize` writes normalized YAML.

## Testing plan
- Unit:
  - `tests/unit/test_project_backend.py` (mode/engine bootstrap shape, legacy normalization, wiring)
- Integration:
  - `tests/integration/test_cli_config.py` (`show-options`, `config show`, `config validate`)
  - `tests/integration/test_cli_smoke.py`
- Regression/golden fixtures:
  - Not required for artifact schema because on-disk run/bundle artifacts are unchanged.

## Rollout / release notes
- Add `CHANGELOG.md` entry describing new execution schema and config CLI.
- Users can migrate existing YAML via `classiflow config normalize --out ...`.
