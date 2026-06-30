# AGENTS.md

## Purpose
This file is the **single operational contract** for humans and coding agents working in this repo.
Before changing code, read:
- `docs/change-management/README.md`
- the module doc(s) for the area you will touch in `docs/change-management/*.md`

`docs/change-management/` is the **canonical source of truth** for module objectives, IO contracts, invariants, and risk. If this file conflicts with code, treat it as a signal to (a) verify behavior in `src/classiflow/`, and (b) update the relevant change-management doc(s) as part of the change.

## Repository Overview
- Language/runtime: Python `>=3.9` (see `pyproject.toml`)
- Package layout: `src/classiflow/`
- CLI entrypoint: `classiflow = classiflow.cli.main:app` (Typer; see `pyproject.toml` and `src/classiflow/cli/main.py`)
- Optional components:
  - Streamlit UI (extra: `classiflow[app]`)
  - UI API (FastAPI) (extra: `classiflow[ui]`)
  - Parquet I/O (extra: `classiflow[parquet]`)

**Confirmed CLI surfaces (do not invent others)**
- Training: `classiflow train-binary`, `classiflow train-meta`, `classiflow train-multiclass`, `classiflow train-hierarchical`
- Inference: `classiflow infer`, `classiflow infer-hierarchical`
- Bundles: `classiflow bundle create|inspect|validate`
- Lineage migration: `classiflow migrate run|batch`
- Projects: `classiflow project init|bootstrap|register-dataset|run-technical|run-feasibility|build-bundle|run-test|recommend|ship`
- Stats: `classiflow stats run|viz|umap` (`umap` is a stub that points to `scripts/umap_plot.py`)
- UI: `classiflow ui serve|reindex|open|init|check`
- Utilities: `classiflow check-compatibility`, `classiflow compare-smote`, `classiflow summarize` (placeholder), `classiflow export-best` (placeholder)

## Prime Directives (Non-Negotiable Invariants)
Synthesized from `docs/change-management/README.md` and module docs (especially `splitting`, `training`, `inference`, `bundles`, `metrics`, `lineage`, `artifacts`, `projects`).

1) No leakage (nested CV + transforms)
- Outer CV: **outer validation must not influence** model selection, preprocessing, feature selection, thresholding, or hyperparameter tuning.
- Inner CV: tuning must run **only on outer-train rows**.
- Any transform (scaler, sampler, feature processing) must be fit **only on the training split** for that fold/split.

2) Patient/group constraints (when enabled)
- If `patient_col`/patient grouping is configured, **no patient ID may appear in both train and validation** for *both outer and inner* splits.
- `splitting.make_group_labels` must fail closed on conflicting patient labels; do not “majority vote” in this module.

3) Determinism / reproducibility
- Same data + same config + same `random_state` ⇒ deterministic split generation and comparable results.
- Seed propagation must be consistent across: splitting, sklearn estimators, SMOTE/adaptive SMOTE, torch training utilities.
- Torch device fallback must be explicit and enforceable via `require_torch_device`.

4) Stable artifact and report schemas (public API)
- Treat on-disk outputs as public API for downstream tools (`inference`, `bundles`, `projects`, `ui_api`, Streamlit UI):
  - `run.json` (training manifest) keys and semantics (`lineage`)
  - metrics CSV filenames and column schemas (`artifacts`, `metrics`)
  - bundle layout and required files (`bundles`)
  - project filesystem layout (`projects`)
- Prefer **additive** changes to schemas. Renames/removals require migration tooling or explicit, well-documented breaking changes.

5) Feature and class ordering contracts
- Feature selection order must be deterministic (notably: sorted numeric auto-detection in `data`).
- In meta/multiclass/hierarchical flows, class ordering artifacts (`classes.csv`, `meta_classes.csv`) are contracts and must remain respected end-to-end.

6) Compatibility boundaries and security boundaries
- `ui_api` artifact allowlisting is a security boundary. Do not broaden it casually.
- Loader tolerance for legacy manifests/artifacts is intentional; if something becomes unsupported, fail loudly with remediation guidance.

## Change Risk Levels & Required Process
Risk levels are defined in `docs/change-management/README.md`. Apply the *highest* risk touched by your change.

### Low risk (refactor; no behavior/output/schema changes)
Required:
- Update/keep `docs/change-management/<module>.md` consistent if code motion changes paths/symbols.
- Run targeted unit tests for touched area (see module table below).

### Medium risk (additive features; backward-compatible default/output changes)
Required:
- Add/adjust tests for new behavior (unit first; integration when IO contracts are involved).
- Update relevant `docs/change-management/<module>.md` sections:
  - Inputs/Outputs
  - Invariants (if strengthened/clarified)
  - Testing requirements (if new tests added)
- Add an entry under `CHANGELOG.md` → `[Unreleased]`.

### High risk (any of: split logic, training/inference semantics, metrics definitions, artifact schema, bundle layout, project layouts, UI API contracts)
Required:
- Write an ADR under `docs/adr/` before/with the change (see `docs/adr/README.md`).
- Add regression/golden tests (fixture-based) for:
  - leakage prevention / group split invariants
  - output schemas (critical CSV columns, manifest keys, bundle required files)
  - backward compatibility (loading prior `run.json` / bundles / artifacts)
- Update `docs/change-management/<module>.md` and add a `CHANGELOG.md` entry.
- Confirm migration/deprecation story (see below).

**High-risk modules (explicitly tagged in `docs/change-management/README.md`)**
- `bundles`, `inference`, `metrics`, `projects`, `splitting`, `training`

## Module Map & Ownership
“Ownership” here means: if you change it, you own updating its change-management doc, tests, and compatibility story.

Risk profile rule used here:
- **High** if tagged `(High risk)` in `docs/change-management/README.md`
- **Medium** otherwise (unless the module doc explicitly states otherwise)

| Module | Objective (1 line) | Primary inputs/outputs (1 line) | Risk | Tests to run |
|---|---|---|---|---|
| `classiflow` | Stable import surface and `__version__`. | Re-exports training/config/task entrypoints; no filesystem IO. | Medium | `python -c "import classiflow; print(classiflow.__version__)"` |
| `config` | Config dataclasses + path resolution. | CLI flags → config JSON via `.save()`; `_resolve_data_path` compat. | Medium | `pytest tests/unit/test_compatibility.py` |
| `cli` | User-facing command contracts. | Typer CLI → training/inference/projects/bundles/ui APIs. | Medium | `pytest tests/integration/test_cli_smoke.py` |
| `io` | Training loaders + compatibility checks. | path + `label_col` → `(X,y[,groups])`; compatibility report. | Medium | `pytest tests/unit/test_compatibility.py` |
| `data` | Unified CSV/Parquet/dataset loading. | `DataSpec`/path → `LoadedDataset` or `pd.DataFrame`. | Medium | `pytest tests/data/test_loaders.py` |
| `splitting` | Group/patient-aware split iterators. | `(df,y,patient_col)` → deterministic `(tr_idx,va_idx)`; leakage asserts. | High | `pytest tests/splitting/test_group_stratified.py` |
| `tasks` | Build OvR/pairwise/composite tasks. | `classes` (+ tasks JSON) → `{task_name: labeler}`. | Medium | `pytest tests/unit/test_tasks.py` |
| `training` | Nested CV training pipelines across modes. | `*Config` → `run.json` + metrics/plots + mode artifacts. | High | `pytest tests/training/test_patient_stratified_wiring.py` |
| `evaluation` | Compare variants (SMOTE vs none) for review. | metrics CSVs → reports/plots. | Medium | `pytest tests/unit/test_smote_comparison.py` |
| `inference` | End-to-end inference for runs/bundles. | run-dir/bundle + data → `predictions.csv` + optional metrics/plots. | High | `pytest tests/inference/test_preprocess.py` |
| `artifacts` | Save/load metrics tables and models. | results dict/joblib → stable filenames + loader compat. | Medium | `pytest tests/bundles/test_bundle_roundtrip.py` |
| `bundles` | Portable ZIP bundles + validation/version checks. | `run_dir` → bundle `.zip` layout + `artifacts.json`. | High | `pytest tests/bundles/test_bundle_roundtrip.py` |
| `lineage` | Hashing + run manifests for provenance. | data + config → `run.json`; compat warnings. | Medium | `pytest tests/lineage/test_manifest.py` |
| `metrics` | Stable metric definitions + scorer order. | labels/scores/probas → metrics dicts + `SCORER_ORDER`. | High | `pytest tests/unit/test_metrics.py` |
| `models` | Estimator registries + SMOTE utilities + torch wrappers. | model specs → estimators/grids; SMOTE safe fallback. | Medium | `pytest tests/unit/test_smote.py` |
| `backends` | Backend/model-set registry (sklearn/torch). | `backend`/`model_set` → estimators + grids; device/seed rules. | Medium | `pytest tests/unit/test_backend_registry.py` |
| `projects` | Project workflows + on-disk layout + promotion. | project tree + thresholds → runs/bundles/recommendations. | High | `pytest tests/integration/test_project_meta_pipeline.py` |
| `plots` | Training/inference plot helpers. | arrays + classes → stable `*.png` files. | Medium | `pytest tests/integration/test_meta_inference_consistency.py` |
| `stats` | Statistical analysis + workbooks/plots. | **CSV input** → `stats_results/` outputs + viz. | Medium | `pytest tests/stats/test_preprocess.py` |
| `streamlit_app` | Thin UI wrapper around APIs. | uploaded CSV → derived artifacts + inference outputs. | Medium | (manual) `streamlit run -m classiflow.streamlit_app.app` |
| `ui_api` | FastAPI backend + artifact allowlisting. | projects tree → API models + safe artifact serving. | Medium | `pytest tests/test_ui_api.py` |
| `validation` | Drift detection + drift reports. | train/inf summaries → drift tables + warnings. | Medium | `pytest tests/validation/test_drift.py` |

## Testing Policy
- Prefer unit tests for pure logic (schemas, hashing, metrics, splits).
- Prefer integration tests when IO contracts are involved:
  - on-disk artifact layouts (`run.json`, metrics CSVs, bundle contents)
  - end-to-end meta inference consistency / alignment
  - project workflow orchestration
- Add regression fixtures when:
  - changing a high-risk module
  - fixing a bug that previously regressed (capture the failure mode)
  - changing output schemas/column orders/manifest keys
- Deterministic tests:
  - pass explicit seeds; assert stable ordering and stable schema invariants
  - use tolerance bands only for stochastic torch training and document why

## Tooling / Quality Gates
Configured in `pyproject.toml`:
- Format: `black .`
- Lint: `ruff check .`
- Types: `mypy src/classiflow`
- Tests: `pytest`

Dev install:
- `pip install -e '.[dev]'`

## Documentation Policy
- `docs/change-management/` is canonical for module behavior and contracts.
- Any change touching a module must update its corresponding doc(s) in `docs/change-management/`.
- High-risk changes require an ADR in `docs/adr/`.

Doc update checklist (for PRs/agent outputs):
- Update module doc Objective/Public Interfaces if symbols/paths changed
- Update Inputs/Outputs and list any new files/columns
- Re-state invariants that guard leakage, determinism, schema stability
- Update Testing Requirements section if new tests are needed/added
- Update `README.md` only when CLI user-facing behavior changes

## Deprecation & Backward Compatibility
- CLI flags and defaults are public API:
  - prefer additive flags; deprecate before removal; keep `--data-csv` path for backward compatibility
- Artifact schemas:
  - additive-first (new fields/files are OK)
  - renames/removals require a migration plan and versioned schema story
- Bundles:
  - required bundle files and paths must remain stable or be migrated
  - keep loader tolerant; if refusing older bundles, fail with actionable guidance

## Release & Versioning
- Versioning: SemVer (see `pyproject.toml` and `src/classiflow/__init__.py`)
- Current version: `0.1.0`
- Bumping:
  - update `pyproject.toml` `project.version`
  - update `src/classiflow/__init__.py` `__version__`
- Changelog:
  - record all user-visible changes under `CHANGELOG.md` → `[Unreleased]`

Release checklist:
- Run `ruff check .`, `black .`, `mypy src/classiflow`, `pytest`
- Verify high-risk invariants (splits/leakage, determinism) did not regress
- Confirm bundle validation and inference on representative fixtures (or documented limitations)
- Update `CHANGELOG.md` and any relevant change-management docs/ADRs

## PR / Agent Output Template
Required in every PR description / agent final output:
- Intent:
- Scope (files):
- User-visible changes:
- Risk level (Low/Medium/High) and why:
- Tests added/run:
- Docs updated (change-management / ADR / README):
- Compatibility / migration notes (artifacts, bundles, CLI):
- Reproducibility notes (seeds, determinism, leakage checks):

