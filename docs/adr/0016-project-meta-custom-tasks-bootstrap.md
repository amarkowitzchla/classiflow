# ADR 0016: Project Bootstrap Support for Meta Custom Tasks

## Status
Accepted

## Context
`classiflow train-meta` already supports `tasks_json` and `tasks_only`, but project workflows did not expose or persist this in `project.yaml`. As a result, users could not configure custom task sets at `project bootstrap` time, and `project run-technical` always used auto OvR/pairwise tasks.  
Relevant module contracts:
- `docs/change-management/projects.md`
- `docs/change-management/projects/orchestrator.md`
- `docs/change-management/training/meta.md`
- `docs/change-management/cli.md`

## Decision
Add additive project-level support for meta custom task configuration:
- New bootstrap CLI flags:
  - `classiflow project bootstrap --tasks-json PATH`
  - `classiflow project bootstrap --tasks-only`
- Persist these in `project.yaml` under:
  - `task.tasks_json`
  - `task.tasks_only`
- Enforce validation rules:
  - `task.tasks_only` requires `task.tasks_json`
  - both keys are valid only when `task.mode=meta`
- In `run_technical_validation`, resolve `task.tasks_json` relative to project root when needed, validate existence, and pass both settings into `MetaConfig` so meta training uses the configured task set.

## Consequences
### Positive
- Project workflows can now express custom meta task definitions without leaving `project` commands.
- Bootstrap-to-technical-validation behavior is consistent with standalone `train-meta`.
- Configuration remains additive and explicit in `project.yaml`.

### Negative / tradeoffs
- `task.tasks_json` persisted as an absolute path from bootstrap can reduce portability if project directories are moved between machines.
- Validation is stricter for misconfigured non-meta projects using task customization keys.

## Compatibility
- CLI flags:
  - Added optional, backward-compatible flags on `project bootstrap`.
- Artifacts/manifests/bundles:
  - No schema changes to training artifacts, bundles, or manifests.
  - Task definitions in produced runs may change when users opt into custom tasks (expected behavior).
- Migrations/deprecations:
  - No migration required for existing projects.

## Testing plan
- Unit:
  - `tests/unit/test_project_backend.py::test_project_bootstrap_meta_persists_custom_tasks`
  - `tests/unit/test_project_backend.py::test_project_bootstrap_tasks_only_requires_tasks_json`
  - `tests/unit/test_project_backend.py::test_project_run_passes_backend_settings` (extended for task wiring)
- Integration:
  - Existing project integration suite remains valid; this change is additive.
- Regression/golden fixtures:
  - Regression coverage added for bootstrap config persistence and orchestration wiring of task settings.

## Rollout / release notes
Add entry under `CHANGELOG.md` `[Unreleased]` in `Changed`:
- `project bootstrap` now supports `--tasks-json/--tasks-only` and `project run-technical` forwards them to meta training.
