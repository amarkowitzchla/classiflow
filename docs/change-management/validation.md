# validation

## Objective
- Detect feature distribution drift between training and inference datasets.
- Provide drift summaries and human-readable warnings for review and UI display.
- Persist drift reports as CSV/Excel for traceability.

## Public Interfaces
- `compute_feature_summary(X: pd.DataFrame, feature_names: list[str]|None = None) -> dict`
- `compute_drift_scores(train_summary: dict, inference_summary: dict) -> pd.DataFrame`
- `detect_drift(drift_df: pd.DataFrame, z_threshold=3.0, missing_threshold=0.1, median_threshold=0.5) -> (flagged_df, warnings)`
- `create_drift_report(drift_df, flagged_features, output_dir, thresholds=None) -> dict[str, Path]`
- `save_feature_summaries(summary, output_path) -> None`
- `load_feature_summaries(input_path) -> dict`
- `create_drift_banner_message(flagged_count, total_features, top_drifted=None) -> str`

## Inputs
- Feature matrices as pandas DataFrames; expects numeric columns for meaningful drift scores.
- Feature names list controls which columns are summarized (defaults to `X.columns`).
- Thresholds:
  - `z_threshold` for mean shifts normalized by training std
  - `missing_threshold` for changes in missingness rate
  - `median_threshold` for median shifts normalized by training IQR

## Outputs
- Drift tables:
  - `feature_drift_summary.csv`
  - `feature_drift_summary.xlsx` with multiple sheets (`Drift`, `Drift_Flagged`, optional `Thresholds`)
- Warnings list suitable for logging or UI banners.
- JSON persistence for summaries via `save_feature_summaries`.

## Internal Workflow
- Summaries compute mean/std/median/quantiles after filtering non-finite values.
- Drift scoring computes:
  - `z_shift`, `missing_delta`, `median_shift` plus absolute variants for ranking.
- Detection selects features exceeding any threshold and sorts by maximum drift.

## Dependencies
- Upstream callers:
  - Inference: manifests include drift warnings; inference modules may call drift checks depending on configuration.
  - UI: can display drift banner message and drift reports as artifacts.
- External dependencies: `pandas`, `numpy`, optional `xlsxwriter`.

## Invariants & Safety Constraints
- Drift detection must be robust to NaNs/infs and non-numeric columns (best-effort, warn rather than crash when possible).
- Report filenames are a contract for reviewers and UI artifact browsing.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Improve summary statistics (additive columns) | Medium | Update tests; ensure Excel writer formatting still works |
| Change thresholds/defaults | Medium | Regression tests; document change in inference warnings semantics |
| Change drift score formulas | High | Add golden tests using fixed fixtures; update reviewer guidance |

## Testing Requirements
- Unit: `pytest tests/validation/test_drift.py`

## Common Pitfalls
- Dividing by zero when training std/IQR is zero (module guards this; keep behavior stable).
- Treating drift scores as definitive performance indicators; they’re diagnostic signals only.
- Writing Excel files without `xlsxwriter` installed (ensure graceful failure if extending).

## Examples
```python
import pandas as pd
from classiflow.validation import compute_feature_summary, compute_drift_scores, detect_drift

train = pd.DataFrame({"f1": [0, 1, 2], "f2": [10, 10, 10]})
inf = pd.DataFrame({"f1": [10, 11, 12], "f2": [10, 10, 10]})

drift = compute_drift_scores(compute_feature_summary(train), compute_feature_summary(inf))
flagged, warnings = detect_drift(drift)
print(warnings[:1])
```

