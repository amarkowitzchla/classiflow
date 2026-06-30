# metrics

## Objective
- Provide consistent metric computation for training and inference outputs.
- Define scorer sets for GridSearchCV so model selection is comparable across runs.
- Provide probability quality metrics (calibration) used in binary/multiclass/meta/hierarchical training and inference reporting.

## Public Interfaces
- Scorers (`src/classiflow/metrics/scorers.py`):
  - `SCORER_ORDER: list[str]`
  - `get_scorers() -> dict[str, Any]`
- Binary metrics (`src/classiflow/metrics/binary.py`):
  - `compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]`
- Probability quality (`src/classiflow/metrics/calibration.py`):
  - `compute_probability_quality(y_true, y_pred, y_proba, classes, bins=10, mode="multiclass", binning="uniform") -> (metrics: dict, curves: dict[str, pd.DataFrame])`
- Calibration policy (`src/classiflow/metrics/calibration_policy.py`):
  - `decide_calibration(...) -> CalibrationDecision`
- Probability-quality diagnostics (`src/classiflow/metrics/probability_quality_checks.py`):
  - `evaluate_probability_quality_checks(...) -> list[ProbQualityRuleResult]`
  - `build_probability_quality_next_steps(...) -> list[str]`
- Decision metrics (`src/classiflow/metrics/decision.py`):
  - `compute_decision_metrics(...) -> dict` (used by inference/meta training)
- Meta binary learner diagnostics (`src/classiflow/metrics/binary_learners.py`):
  - `persist_binary_ovr_fold_outputs(...) -> Path | None`
  - `evaluate_binary_learner_health(...) -> dict`
  - `build_binary_learner_warnings(...) -> list[dict]`

## Inputs
- `compute_binary_metrics`:
  - `y_true` must be binary-coded (0/1)
  - `scores` may be probabilities `[0,1]` or decision values (thresholding differs)
- `get_scorers`:
  - returns sklearn `make_scorer` objects; `"ROC AUC"` uses `response_method="predict_proba"`.
- `compute_probability_quality`:
  - `y_true` and `y_pred` are lists/arrays of class labels (strings expected)
  - `y_proba` shape `(n_samples, n_classes)` aligned to `classes` order
  - `bins` controls ECE binning (`np.linspace(0,1,bins+1)`)
  - `mode` must be one of `binary|multiclass|meta|hierarchical`
  - `binning` controls bin construction (`uniform` or `quantile`)

## Outputs
- Metrics dicts used as row fragments in CSV outputs:
  - binary: includes `accuracy`, `balanced_accuracy`, `precision/ppv`, `recall/sensitivity`, `specificity`, `npv`, `f1`, `mcc`, `roc_auc`
  - calibration: includes `brier_recommended`, `log_loss`, `ece_top1`, plus mode-specific keys:
    - binary: `brier_binary`, `ece_binary_pos`
    - multiclass/meta/hierarchical: `brier_multiclass_sum`, `brier_multiclass_mean`, `ece_ovr_macro`, per-class `ece_ovr__<class>`
    - alignment: `pred_alignment_mismatch_rate`, `pred_alignment_note`
  - compatibility aliases retained for one release: `brier` -> `brier_recommended`, `ece` -> `ece_top1`
  - calibration curves are returned as a map (`top1`, `binary_pos`, and per-class `ovr_<class>`)
  - probability-quality diagnostics return rule records with severity, measured values, thresholds,
    artifact evidence pointers, and actionable recommendations
  - hierarchical-specific rule `PQ-H001` tracks argmax-probability vs postprocessed label mismatch
- `SCORER_ORDER` drives column ordering in:
  - `metrics_inner_cv_splits.csv` / `.xlsx` and related outputs.

## Internal Workflow
- Binary thresholding rule:
  - scores in `[0,1]` ⇒ threshold `0.5`
  - otherwise ⇒ threshold `0.0` (decision_function style)
- Calibration metrics:
  - Top-1 calibration (`ece_top1`) is always defined as confidence of `argmax(y_proba)` vs argmax correctness.
  - Binary mode additionally computes probability calibration on positive-class probability (`ece_binary_pos`).
  - Multiclass-like modes additionally compute OVR probability calibration (`ece_ovr_macro` + per-class OVR components).
  - Multiclass Brier is reported as both unnormalized (`brier_multiclass_sum`) and normalized (`brier_multiclass_mean`, recommended).
  - Quantile binning handles small-N and repeated confidence values by collapsing duplicate quantile edges.
  - Non-probability inputs (scores outside `[0,1]`) are rejected for probability-quality metrics.
- Probability-quality checks:
  - binary mode uses positive-class reliability occupancy (`binary_pos`) and binary-specific regression metrics (`brier_binary`, `ece_binary_pos`)
  - multiclass/meta/hierarchical use top1 occupancy plus OVR diagnostics
  - hierarchical additionally emits mismatch diagnostics (`pred_alignment_mismatch_rate`) and `PQ-H001`
- Calibration decision policy:
  - R1 underconfidence + high top1 accuracy => disable
  - R2 near-perfect top1 accuracy => disable
  - R3 insufficient sample support => disable
  - R4 retain calibration only when Brier improves and log-loss/OVR-ECE do not regress beyond configured limits
- Binary learner health diagnostics:
  - computes per-class OVR metrics by fold and aggregated summaries for meta mode
  - emits rule-based warnings `BL-001`..`BL-006` with evidence paths and recommendations
  - renders base-OVR ROC plots and OVO AUC matrix (derived from final meta probabilities)

## Dependencies
- Upstream callers:
  - Training: `training/nested_cv.py`, `training/meta.py`, `training/multiclass.py`
  - Inference: `inference/metrics.py` composes classification metrics and uses calibration helpers
  - Artifacts: `artifacts/saver.py` uses `SCORER_ORDER`
- External dependencies: `numpy`, `pandas`, `sklearn`.

## Invariants & Safety Constraints
- Metric definitions are a public contract for:
  - promotion gates (`projects/promotion.py` and thresholds configs)
  - reviewer-facing outputs and SMOTE comparisons
  - regression tests asserting stability
- `SCORER_ORDER` is a schema contract for split-metrics exports.
- Class/probability alignment must remain consistent:
  - `classes` ordering must match `y_proba` columns; otherwise calibration metrics are invalid.

## Change Risk Guide
| Change Type | Risk Level | Required Actions |
|---|---|---|
| Add a new metric (additive) | Medium | Update tests and any reporting tables; document new output columns |
| Change an existing metric’s formula/thresholding | High | Regression/golden tests; update docs; validate downstream gate thresholds |
| Rename metric labels used in CSV columns or `SCORER_ORDER` | High | Migration plan; update UI/project/promotion and evaluation modules |

## Testing Requirements
- Unit: `pytest tests/unit/test_metrics.py`
- Unit: `pytest tests/metrics/test_calibration.py`
- Unit: `pytest tests/metrics/test_probability_quality_checks.py`
- Unit: `pytest tests/metrics/test_binary_learners.py`
- Integration: `pytest tests/integration/test_meta_brier_alignment.py`
- Training calibration: `pytest tests/training/test_meta_calibration.py`
- Unit policy rules: `pytest tests/metrics/test_calibration_policy.py`
- Integration artifacts: `pytest tests/training/test_meta_calibration_policy_integration.py`

## Common Pitfalls
- Mixing label encodings (int vs str) across modules; calibration expects class labels as strings.
- Using ROC AUC on degenerate splits (single-class); functions guard with `NaN` but downstream code must tolerate.
- Changing default bin counts for ECE impacts comparisons and promotion thresholds.

## Examples
```python
import numpy as np
from classiflow.metrics import compute_binary_metrics

y_true = np.array([0, 1, 0, 1])
scores = np.array([0.1, 0.9, 0.8, 0.7])
print(compute_binary_metrics(y_true, scores)["roc_auc"])
```

## High-Risk Change Protocol
- Required design note (ADR):
  - Specify the metric definition change, intended impact, and why it is correct.
  - Document expected shifts in historical baselines and promotion thresholds.
- Required test additions:
  - Add a regression fixture demonstrating the prior bug/behavior and expected new behavior.
  - Add a “class/proba alignment” test for any metric using `y_proba`.
- Required backward compatibility checks:
  - Ensure CSV column names used by `projects/` and `evaluation/` are unchanged or migrated.
  - Ensure SMOTE comparison still detects metrics columns correctly.
- Required release note items:
  - “Metric definition changes” section including migration guidance for thresholds and comparisons.
