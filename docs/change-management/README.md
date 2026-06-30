# Change Management (Pre-Change Reference)

This directory is a **pre-change reference** for the `classiflow` Python package (`src/classiflow`). Before modifying any module, read the relevant module doc(s) here and follow the checklists and risk protocol.

## How To Use These Docs
- Start with the module you intend to change (links below), then read any linked sub-docs.
- Treat **artifacts, manifests, and schemas** as public API: changes must be intentional, versioned, and migration-ready.
- **Nested CV correctness and leakage prevention are non-negotiable**: any change touching splitting, training, scoring, or inference must preserve leakage constraints.

## Change Risk Levels

| Risk | Definition | Required Actions |
|---|---|---|
| Low | Pure refactor; no behavior/output/artifact/schema changes | Unit tests for touched code; no output diffs; update docstrings if needed |
| Medium | Adds functionality or changes defaults/outputs in a backwards-compatible way | Add/adjust tests; update docs; confirm deterministic behavior; add compatibility notes |
| High | Changes split logic, training/inference semantics, metric definitions, artifact schema, bundle contents, or manifests | ADR/design note; golden/regression tests; explicit migration plan; backwards compatibility checks; release note entry |

## Pre-Change Checklist (Agents & Developers)
- Identify the **public surface** affected: CLI flags, exported symbols, config fields, artifact filenames, manifest keys, bundle layout.
- Enumerate **data contracts**: required columns, dtypes, feature ordering, label encoding, patient/group stratification rules.
- Confirm **determinism** expectations: `random_state`/seed propagation, torch device fallback behavior, CV split reproducibility.
- Confirm **no leakage**: groups/patients never cross train/val in outer or inner CV; transforms are fit only on training splits.
- Update **tests** (and add regression/golden tests if this is Medium/High risk).
- For High risk: write a short design note, update compatibility/migration guidance, and ensure old artifacts/bundles still load or fail loudly with guidance.

## Module Index

### Root Modules
- `docs/change-management/classiflow.md` — package exports (`classiflow/__init__.py`)
- `docs/change-management/config.md` — config dataclasses and path resolution (`classiflow/config.py`)

### Package Directories
- `docs/change-management/artifacts.md`
- `docs/change-management/backends.md`
- `docs/change-management/bundles.md` *(High risk)*
- `docs/change-management/cli.md`
- `docs/change-management/data.md`
- `docs/change-management/evaluation.md`
- `docs/change-management/inference.md` *(High risk)* — see `docs/change-management/inference/`
- `docs/change-management/io.md`
- `docs/change-management/lineage.md`
- `docs/change-management/metrics.md` *(High risk)*
- `docs/change-management/models.md`
- `docs/change-management/plots.md`
- `docs/change-management/projects.md` *(High risk)* — see `docs/change-management/projects/`
- `docs/change-management/splitting.md` *(High risk)*
- `docs/change-management/stats.md`
- `docs/change-management/streamlit_app.md`
- `docs/change-management/tasks.md`
- `docs/change-management/training.md` *(High risk)* — see `docs/change-management/training/`
- `docs/change-management/ui_api.md`
- `docs/change-management/validation.md`

