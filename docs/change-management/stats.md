# stats

## Objective
- Provide a publication-oriented statistical analysis pipeline for feature comparisons across classes.
- Produce reproducible Excel workbooks and plots suitable for reviewer-facing outputs.
- Keep stats outputs stable for downstream “project” workflows and UI browsing.

## Public Interfaces
- `run_stats(...) -> dict[str, Any]` in `src/classiflow/stats/api.py`
- `run_stats_from_config(config: StatsConfig, ...) -> dict[str, Any]` in `src/classiflow/stats/api.py`
- `run_visualizations(...) -> dict[str, Any]` in `src/classiflow/stats/api.py` (exported via `stats/__init__.py`)
- Configs:
  - `StatsConfig` in `src/classiflow/stats/config.py`
  - `VizConfig` in `src/classiflow/stats/config.py`
- CLI:
  - `classiflow stats run ...` and `classiflow stats viz ...` in `src/classiflow/cli/stats.py`

## Inputs
- Primary input is a **CSV path** (read via `pd.read_csv` in `stats/api.py`).
- Note: `classiflow stats` CLI accepts `--data` with Parquet paths via `_resolve_data_path`, but the current stats implementation still reads via `pd.read_csv`; treat Parquet support as not implemented unless/until the stats API is updated to use `classiflow.data.load_table`.
- Required columns:
  - `label_col` must exist
  - Features are selected as numeric columns after preprocessing (see `stats/preprocess.py`)
- Config parameters:
  - `alpha`, `min_n`, `dunn_adjust`
  - optional `classes` subset/order
  - `feature_whitelist`/`feature_blacklist`
  - fold-change configuration (`fc_center`, `fc_eps`) used in reports/viz

## Outputs
- Directory layout (under `outdir / "stats_results"`):
  - `publication_stats.xlsx` (via `write_publication_workbook`)
  - optional legacy outputs:
    - `stats_results.xlsx`
    - multiple `*.csv` tables (normality, parametric, nonparametric results)
- Visualization outputs (under `outdir / "viz"` by default):
  - boxplot grids, volcano plots, fold-change plots, heatmaps, and CSV summaries (see `stats/viz.py`)

## Internal Workflow
- `run_stats_from_config` pipeline (high level):
  - load CSV
  - `prepare_data(...)` selects features/classes and handles allow/deny lists
  - normality tests (Shapiro–Wilk)
  - **binary (2-class)**:
    - Welch t-test if both classes pass normality and `min_n`
    - Mann–Whitney U otherwise
    - per-feature p-value adjustment across features (method=`dunn_adjust`)
  - **multiclass (3+)**:
    - parametric tests (Welch t-test / ANOVA + Tukey)
    - nonparametric tests (Kruskal–Wallis + Dunn)
  - build publication tables (`stats/reports.py`)
  - write workbook(s) and optional CSVs (`stats/excel.py`)
- Visualization pipeline uses stats outputs when available (Dunn p-values, pairwise summaries).

## Dependencies
- Upstream callers:
  - CLI: `src/classiflow/cli/stats.py`
  - Streamlit: `streamlit_app/pages/02_Statistics.py`
  - Projects: feasibility workflows call stats APIs (see `projects/orchestrator.py`)
- External dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `openpyxl`/`xlsxwriter` depending on writer.

## Invariants & Safety Constraints
- Output filenames and workbook sheet schemas are used by downstream workflows; treat as public API for projects and reviewers.
- `classes` order must be preserved in reports (affects pairwise comparisons and plots).
- Multiple testing correction semantics (`dunn_adjust`) must not silently change.
  - Binary mode uses `dunn_adjust` for per-feature p-value adjustment across features.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new report sheets/plots (additive) | Medium | Update docs; ensure outputs are deterministic; add tests where feasible |
| Change statistical test definitions or defaults | High | Add regression tests; document methodological changes; update release notes |
| Change file naming/layout under `stats_results/` | High | Migration guidance; update UI/project consumers; golden output tests |

## Testing Requirements
- Unit: `pytest tests/stats/test_preprocess.py`
- Unit: `pytest tests/stats/test_normality.py`
- Unit: `pytest tests/stats/test_effects.py`
- Unit: `pytest tests/stats/test_binary.py`

## Common Pitfalls
- Large feature counts can create heavy workbooks; be cautious about memory/time.
- Using CSV-only ingestion here while other subsystems support Parquet; avoid inconsistent behavior when extending.
- Misinterpreting `alpha` scope (per-test vs FDR-corrected contexts).

## Examples
```bash
classiflow stats run --data data.csv --label-col diagnosis --outdir derived
classiflow stats viz --data data.csv --label-col diagnosis --outdir derived
```
