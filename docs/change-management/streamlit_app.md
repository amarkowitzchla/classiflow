# streamlit_app

## Objective
- Provide a lightweight Streamlit UI for interactive training, stats, and inference workflows.
- Surface outputs from `derived/` for quick inspection during local iteration.
- Offer a user-facing wrapper around library APIs without adding new ML logic.

## Public Interfaces
- Streamlit entrypoints (module-level scripts):
  - `src/classiflow/streamlit_app/app.py`
  - Pages in `src/classiflow/streamlit_app/pages/`:
    - `01_Train_Models.py`
    - `02_Statistics.py`
    - `06_Inference.py`
- Helpers:
  - `list_outputs(derived: Path) -> dict[str, list[str]]` in `src/classiflow/streamlit_app/ui/helpers.py`
  - `use_theme()` in `src/classiflow/streamlit_app/ui/style.py`

## Inputs
- User-uploaded CSV files (Streamlit upload widgets).
- Assumes local working directory has (or can create):
  - `data/`
  - `derived/` (and sometimes `derived_hierarchical/`)
- Uses library configs:
  - `TrainConfig`, `MetaConfig` (`01_Train_Models.py`)
  - `InferenceConfig` (`06_Inference.py`)

## Outputs
- Writes uploaded files to `data/` or `data/uploaded/`.
- Writes training artifacts to `derived/` via library calls.
- Writes inference outputs to `inference_output/<fold>/` via `run_inference`.

## Internal Workflow
- The UI is a thin orchestration layer:
  - read/upload CSV
  - pick label columns and options
  - call `classiflow.training.*` or `classiflow.stats.*` or `classiflow.inference.run_inference`
  - render resulting files and tables

## Dependencies
- Upstream callers: humans (Streamlit UI).
- Downstream calls:
  - training: `classiflow.training.train_binary_task`, `classiflow.training.train_meta_classifier`
  - stats: `classiflow.stats.run_stats`, `classiflow.stats.run_visualizations`
  - inference: `classiflow.inference.run_inference`
- External dependencies: `streamlit`, `pandas`.

## Invariants & Safety Constraints
- The UI must not introduce new “hidden defaults” that diverge from CLI/library behavior.
- Output directory conventions should remain aligned with library-generated artifacts (so `list_outputs` remains accurate).

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| UI-only layout/text changes | Low | Manual smoke check in Streamlit |
| Change how configs are constructed (defaults) | Medium | Ensure parity with CLI/library; update user-facing text |
| Change output directory conventions | Medium | Update `list_outputs` and documentation; avoid breaking existing runs |

## Testing Requirements
- No dedicated Streamlit tests currently; rely on integration tests for underlying APIs.
- Suggested manual smoke:
  - `streamlit run -m classiflow.streamlit_app.app`

## Common Pitfalls
- Writing uploaded data to conflicting filenames (use safe names and per-session folders if extending).
- Assuming only CSV; the library supports Parquet for CLI flows.
- Import path hacks (see `06_Inference.py`) can mask packaging issues; keep module imports clean when possible.

## Examples
```bash
streamlit run -m classiflow.streamlit_app.app
```

