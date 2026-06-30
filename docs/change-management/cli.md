# cli

## Objective
- Provide the `classiflow` command-line interface for training, inference, bundling, migration, project workflows, stats, and UI serving.
- Define user-facing flags and help text that map into config dataclasses and module APIs.

## Public Interfaces
- Entry point:
  - `app = typer.Typer(...)` in `src/classiflow/cli/main.py`
  - Package entrypoint configured as `classiflow = classiflow.cli.main:app`
- Top-level commands in `src/classiflow/cli/main.py` (Typer):
  - `train-binary`
  - `train-meta`
  - `train-multiclass`
  - `check-compatibility`
  - `train-hierarchical`
  - `infer-hierarchical`
  - `infer`
  - `compare-smote`
- Subcommand groups added in `src/classiflow/cli/main.py`:
  - `classiflow stats ...` (`src/classiflow/cli/stats.py`)
  - `classiflow bundle ...` (`src/classiflow/cli/bundle.py`)
  - `classiflow migrate ...` (`src/classiflow/cli/migrate.py`)
  - `classiflow project ...` (`src/classiflow/cli/project.py`)
  - `classiflow config ...` (`src/classiflow/cli/config.py`)
  - `classiflow ui ...` (`src/classiflow/cli/ui.py`)

## Inputs
- CLI inputs include:
  - dataset path (`--data` preferred; legacy `--data-csv`)
  - label columns (`--label-col`, `--label-l1`, `--label-l2`)
  - leakage controls (`--patient-col`)
  - reproducibility (`--random-state`)
  - backend/device selection (`--backend`, `--device`, `--model-set`)
  - output directories (`--outdir`)
- Project CLI consumes project directory trees and YAML config files (see `projects/` docs).
  - `classiflow project bootstrap` supports:
    - `--mode`
    - `--engine sklearn|torch|hybrid`
    - `--device auto|cpu|cuda|mps` (torch/hybrid only)
    - `--tasks-json PATH` (meta mode only)
    - `--tasks-only` (meta mode only; requires `--tasks-json`)
    - `--show-options`
    - `--gate-profile balanced|f1|sensitivity`
    to initialize promotion gate defaults in `registry/thresholds.yaml`.
  - `classiflow project bootstrap|recommend` support:
    - `--promotion-gate-template <id>`
    - `--list-promotion-gate-templates`
  - `classiflow config` supports:
    - `show --mode ... --engine ...`
    - `explain path.to.field`
    - `validate project.yaml`
    - `normalize project.yaml --out ...`

- Stats note:
  - While `classiflow stats ...` accepts `--data` and will resolve Parquet paths, the current stats implementation reads via `pd.read_csv(...)` and therefore effectively requires CSV input (see `docs/change-management/stats.md` and `src/classiflow/stats/api.py`).

## Outputs
- Training/inference outputs:
  - written under user-specified `--outdir` (defaults `derived/` or `derived_hierarchical/`)
  - include metrics CSVs, plots, manifests, and model artifacts depending on mode
- Bundles:
  - `.zip` bundle files created by `classiflow bundle create`
- Migration:
  - `run.json` created/overwritten from `run_manifest.json` for legacy runs
- UI:
  - starts a server and may create `.classiflow/ui.db` (sqlite)

## Internal Workflow
- CLI is primarily orchestration:
  - parse options
  - build `TrainConfig` / `MetaConfig` / `MulticlassConfig` / `HierarchicalConfig` / `InferenceConfig`
  - run compatibility checks for meta/hierarchical training (`io/compatibility.py`)
  - call module APIs

## Dependencies
- Upstream callers: end users.
- Downstream calls:
  - training (`classiflow.training.*`)
  - inference (`classiflow.inference.run_inference`, `classiflow.inference.HierarchicalInference`)
  - bundles (`classiflow.bundles.*`)
  - evaluation (`classiflow.evaluation.smote_comparison.SMOTEComparison`)
  - ui (`classiflow.ui_api.server`)
- External dependencies: `typer`, `click` (legacy command exists in `cli/infer.py` but Typer command in `cli/main.py` is primary).

## Invariants & Safety Constraints
- CLI flag semantics are public API; changes must be deliberate and documented.
- `--data` vs `--data-csv` resolution must remain backward compatible (`config._resolve_data_path`).
- Patient-level stratification is a safety feature; the CLI must not “auto-enable” it silently.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new optional flag (additive) | Medium | Add tests; update docs and `--help` examples |
| Change default behavior/output paths | High | Regression tests; migration/compat notes; avoid silent changes |
| Remove/rename commands/flags | High | Deprecation plan; update docs; update project templates |

## Testing Requirements
- CLI-facing behavior is covered indirectly via integration tests and module tests.

## Change Log
- **Production-readiness cleanup (2026-06-30)** — Low risk:
  - `cli/backfill.py`: Removed dead variables (`fold_data`, `fold_num`, `metrics_file`,
    `proba_file`) that were assigned in the fold-scan loop but never consumed. No behavior change.
  - `cli/infer.py`: Removed unused `bundle_loader = None` sentinel. No behavior change.
  - `cli/main.py`: Dropped unused return-value binding from `train_binary_task()`. No behavior change.
  - All CLI files reformatted with `black` and `ruff --fix` (import ordering, f-string cleanup).
    No logic changed.
- Suggested: run focused tests for affected area, e.g.:
  - `pytest tests/unit/test_compatibility.py`
  - `pytest tests/bundles/test_bundle_roundtrip.py`

## Common Pitfalls
- Adding a new command but not wiring it into `app` or subcommand groups.
- Diverging CLI defaults from library defaults, causing inconsistent behavior across entrypoints.
- Examples in help text must remain accurate; verify after changes.

## Examples
```bash
classiflow --version
classiflow train-meta --data data.parquet --label-col diagnosis --outdir derived --smote both
classiflow infer --data test.parquet --run-dir derived --outdir inference_results --label-col diagnosis
classiflow stats run --data data.csv --label-col diagnosis --outdir derived
```
