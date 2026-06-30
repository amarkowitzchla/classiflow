# plots

## Objective
- Provide consistent plotting utilities for ROC/PR curves, confusion matrices, averaged fold curves, and feature importance visuals.
- Keep plot generation logic centralized so training/inference modules can produce consistent artifacts.

## Public Interfaces
- From `src/classiflow/plots/__init__.py` (implemented in `src/classiflow/plots/hierarchical.py`):
  - `plot_roc_curve(y_true, y_proba, classes, title, save_path, ...)`
  - `plot_pr_curve(y_true, y_proba, classes, title, save_path, ...)`
  - `plot_averaged_roc_curves(all_fpr, all_tpr, all_aucs, title, save_path, ...)`
  - `plot_averaged_pr_curves(all_rec, all_prec, all_aps, title, save_path, ...)`
  - `plot_confusion_matrix(y_true, y_pred, classes, title, save_path, normalize="true", ...)`
  - `plot_feature_importance(importances, feature_names, title, save_path, ...)`
  - `extract_feature_importance_mlp(model, X, y, feature_names, n_permutations=5, ...)`

## Inputs
- Arrays:
  - `y_true` numeric-coded (0..K-1) for multiclass plots; for binary, treated as 0/1.
  - `y_proba` shape `(n_samples, n_classes)`; binary plots expect 2 columns.
- `save_path` is a filesystem `Path` and parent directories must exist (callers handle).

## Outputs
- Image files written to disk:
  - `*.png` plots saved with high DPI (`dpi=300`) and tight bounding boxes.

## Internal Workflow
- Compute curves using sklearn helpers (`roc_curve`, `precision_recall_curve`, `auc`, `average_precision_score`).
- For averaged plots:
  - interpolate fold curves to a common x-axis; plot mean and std band.

## Dependencies
- Upstream callers:
  - Training: `training/nested_cv.py`, `training/meta.py`, `training/multiclass.py`, `training/hierarchical_cv.py`
  - Inference: `inference/plots.py` composes inference-time plots (separate module)
- External dependencies: `matplotlib`, `numpy`, `sklearn`.

## Invariants & Safety Constraints
- Plot outputs are used as reviewer-facing artifacts and UI-viewable files; filenames matter in downstream UIs (e.g., `ui_api` scanner allowlists images).
- Multiclass handling must respect class ordering from training manifests/`classes.csv`.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Refactor plotting internals without output changes | Low | Visual regression spot-check; unit tests if present |
| Change plot filenames or directory conventions | Medium | Update downstream consumers (UI scanners, docs) |
| Change curve computation semantics | High | Add regression tests using known fixtures; document change |

## Testing Requirements
- Indirect coverage via training/inference integration tests:
  - `pytest tests/integration/test_meta_inference_consistency.py`

## Change Log
- **Production-readiness cleanup (2026-06-30)** — Low risk:
  - `plots/schemas.py`: Pydantic `class Config` inner classes converted to `model_config =
    ConfigDict(...)` (Pydantic V2 migration). All schema shapes and JSON serialisation are
    unchanged; only the deprecation warning is removed.
  - All plots files reformatted with `black` and `ruff --fix`. No logic changed.

## Common Pitfalls
- Passing unaligned `y_proba` columns vs `classes` ordering (plots become misleading).
- Plotting when a class has only one label in a split (sklearn may error; code guards some cases).
- Writing plots to non-existent directories (callers should create parents).

## Examples
```python
from pathlib import Path
import numpy as np
from classiflow.plots import plot_confusion_matrix

plot_confusion_matrix(
    y_true=np.array([0, 1, 0, 1]),
    y_pred=np.array([0, 1, 1, 1]),
    classes=["0", "1"],
    title="Example CM",
    save_path=Path("cm_example.png"),
)
```

