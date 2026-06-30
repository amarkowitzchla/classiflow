# evaluation

## Objective
- Provide analysis utilities that compare training variants (notably SMOTE vs no-SMOTE) to support reviewer-facing justification.
- Generate statistical comparisons, overfitting signals, and publication-ready plots.

## Public Interfaces
- `SMOTEComparison` and `SMOTEComparisonResult` in `src/classiflow/evaluation/smote_comparison.py`
  - `SMOTEComparison.from_directory(result_dir, model_type=None, metric_file=...) -> SMOTEComparison`
  - `SMOTEComparison.generate_report(...) -> SMOTEComparisonResult`
  - `SMOTEComparison.save_report(result, outdir) -> dict[str, Path]`
  - `SMOTEComparison.create_all_plots(outdir) -> dict[str, Path]`
- Plotting helpers in `src/classiflow/evaluation/smote_plots.py`
  - `create_all_plots(...)` and underlying plot functions (`plot_delta_bars`, `plot_identity_scatter`, etc.)
- CLI:
  - `classiflow compare-smote ...` in `src/classiflow/cli/main.py`

## Inputs
- Expects training output directories where both SMOTE and no-SMOTE results exist (typically `--smote both`):
  - metric CSV candidates:
    - `metrics_outer_meta_eval.csv`
    - `metrics_outer_multiclass_eval.csv`
    - `metrics_outer_binary_eval.csv`
    - `metrics_outer_eval.csv` (hierarchical)
- Requires a `fold` column or fold subdirectories that can be merged.
- Uses p-values/effect sizes thresholds from CLI arguments (`significance_level`, `min_effect_size`, etc.).

## Outputs
- Human-readable report text and machine-readable JSON:
  - `smote_comparison_YYYYMMDD_HHMMSS.txt`
  - `smote_comparison_YYYYMMDD_HHMMSS.json`
  - summary CSV(s)
- Publication-quality PNG plots (delta bars, identity grids, distributions, trajectories).

## Internal Workflow
- Load SMOTE vs no-SMOTE metrics for matched folds.
- Infer comparable metric columns by dtype and exclude identifiers.
- Compute:
  - mean deltas
  - paired t-tests and Wilcoxon tests (when applicable)
  - Cohen’s d effect sizes
  - heuristic “overfitting detected” based on concurrent drops across metrics
- Plot using seaborn/matplotlib with publication defaults.

## Dependencies
- Upstream callers:
  - CLI `compare-smote`
  - Projects workflows optionally call comparison utilities
- External dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.

## Invariants & Safety Constraints
- Metric column identification must remain stable enough to compare across runs; be cautious when renaming metrics columns in training outputs.
- Statistical test behavior must be explicit (paired vs unpaired assumptions).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add new plots/metrics (additive) | Medium | Add unit tests; ensure CLI output list remains accurate |
| Change recommendation heuristics | High | Regression tests using fixed fixtures; document methodological change |
| Change expected metric file names | High | Update CLI defaults, docs, and any project automation |

## Testing Requirements
- Unit: `pytest tests/unit/test_smote_comparison.py`

## Common Pitfalls
- Comparing mismatched folds (must ensure fold alignment across variants).
- Treating improvements on train phase as meaningful without checking validation phase.
- Plotting with too few folds (tests may be underpowered; code should degrade gracefully).

## Examples
```bash
classiflow compare-smote derived --outdir smote_analysis --model-type meta --metric-file metrics_outer_meta_eval.csv
```

