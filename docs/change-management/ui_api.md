# ui_api

## Objective
- Provide a FastAPI backend for browsing projects/runs/artifacts and recording comments/reviews.
- Normalize on-disk project structures into stable API response models (“Gold layer”).
- Enforce artifact allowlisting for safe serving of files.

## Public Interfaces
- Server runners:
  - `run_server(...)` and `run_dev_server(...)` in `src/classiflow/ui_api/server.py`
- App factory:
  - `create_app(config: UIConfig|None = None) -> FastAPI` in `src/classiflow/ui_api/app.py`
  - `mount_static(app, static_dir)` in `src/classiflow/ui_api/app.py`
- Config:
  - `UIConfig`, `StorageMode` in `src/classiflow/ui_api/config.py`
- Pydantic response models (exported from `src/classiflow/ui_api/__init__.py`):
  - `ProjectCard`, `ProjectDashboard`, `RunDetail`, `Artifact`, `Comment`, `Review`, `Phase`, `DecisionBadge`, etc.
- Local scanning and repositories:
  - `LocalFilesystemScanner` in `src/classiflow/ui_api/scanner.py`
  - Repository interfaces in `src/classiflow/ui_api/repositories/interfaces.py`
  - Local repo in `src/classiflow/ui_api/repositories/local.py`
  - SQLite comment/review store in `src/classiflow/ui_api/repositories/sqlite.py`
- Adapters:
  - `parse_run_manifest`, `parse_metrics` in `src/classiflow/ui_api/adapters/manifest.py`
  - project config/registry parsing in `src/classiflow/ui_api/adapters/project.py`

## Inputs
- Filesystem project tree rooted at `projects_root`:
  - project directories containing `project.yaml`, `runs/<phase>/<run_id>/...`
- Run manifests:
  - `run.json` preferred; legacy `run_manifest.json` may exist.
- Artifact files:
  - served only if extensions are in `ALLOWED_EXTENSIONS` and marked viewable by `VIEWABLE_EXTENSIONS` (scanner rules).
- Comments/reviews:
  - stored in SQLite at `UIConfig.db_path` (default `.classiflow/ui.db`).

## Outputs
- HTTP API responses:
  - `/api/projects`, `/api/projects/{id}`, `/api/runs/{run_key}`, `/api/artifacts/{artifact_id}`, etc.
- Static file serving:
  - optional SPA assets if `static_dir` provided.
- SQLite side effects:
  - DB file creation/migrations for comments/reviews repository.

## Internal Workflow
- `create_app`:
  - validate config
  - choose repository implementations (currently local filesystem + sqlite)
  - register routes
- Local scanning:
  - parse configs/manifests/metrics into normalized structures
  - enumerate artifacts and classify by extension/kind
  - build stable IDs for artifacts and run keys (`project:phase:run_id`)

## Dependencies
- Upstream callers:
  - CLI: `classiflow ui serve` / `classiflow ui reindex` / `classiflow ui check`
- Downstream calls:
  - adapters for manifests/projects (depend on stable on-disk schemas from `projects/` module)
- External dependencies: `fastapi`, `uvicorn`, `pydantic`, `starlette`.

## Invariants & Safety Constraints
- Artifact allowlisting is a security boundary; do not broaden it casually.
- API response models are a contract for the frontend; field renames/removals are breaking.
- Parsing must tolerate legacy runs and incomplete artifacts, returning best-effort summaries.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new API fields (additive) | Medium | Update frontend; add unit tests for adapters |
| Change artifact allowlists or serving behavior | High | Security review; add tests for path traversal and MIME handling |
| Change run/project parsing rules | High | Regression tests against fixture projects; update UI docs |

## Testing Requirements
- Unit/integration: `pytest tests/test_ui_api.py`
- Unit: `pytest tests/unit/test_manifest_adapter.py`

## Change Log
- **Production-readiness cleanup (2026-06-30)** — Low risk:
  - `ui_api/scanner.py`: Added `Any` to typing imports to resolve F821 undefined-name in the
    `_numeric_metrics` local function annotation. No behavior change.

## Common Pitfalls
- Serving files outside the run directory (ensure relative-path enforcement remains intact).
- Breaking on legacy manifest formats; keep adapter compatibility.
- Over-eager caching that misses updated files (scanner has caching; reindex must work).

## Examples
```bash
classiflow ui serve --projects-root ./projects --dev
```

