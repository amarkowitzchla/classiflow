# Technical Validation Report

## Summary Metrics
| metric              |     value |
|:--------------------|----------:|
| f1_macro            | 0.991475  |
| balanced_accuracy   | 0.990741  |
| recall              | 0.990741  |
| specificity         | 0.996032  |
| ppv                 | 0.993056  |
| npv                 | 0.996528  |
| precision           | 0.993056  |
| brier_calibrated    | 0.0218379 |
| ece_calibrated      | 0.213387  |
| log_loss_calibrated | 0.257075  |

## Probability Quality Checks (ECE/Brier)
These checks are diagnostic and not promotion blockers by default. `ece_top1` evaluates Top-1 confidence calibration; `ece_ovr_macro` evaluates one-vs-rest class probability calibration.

Diagnostic plots generated:
- `probability_quality_plots/prob_quality_reliability_top1.png` - Top-1 reliability and occupancy thresholds (PQ-001).
- `probability_quality_plots/prob_quality_confidence_gap.png` - Confidence-gap diagnostics with PQ-002/PQ-003 thresholds.
- `probability_quality_plots/prob_quality_calibration_deltas.png` - Calibration deltas against allowed regressions (PQ-004).

| Severity   | Rule ID   | Finding                                  | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Recommended next action                                                                       |
|:-----------|:----------|:-----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| WARN       | PQ-001    | Low calibration-curve occupancy          | runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_curve_top1_uncalibrated.csv, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold2/binary_none/calibration_curve_top1_uncalibrated.csv, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold3/binary_none/calibration_curve_top1_uncalibrated.csv, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_metadata.json | Interpret ECE qualitatively; inspect reliability plot.                                        |
| WARN       | PQ-002    | Underconfidence behavior detected        | runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Model is conservative; this is usually safe.                                                  |
| WARN       | PQ-004    | Calibration worsened probability quality | runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Disable calibration or use policy auto mode.                                                  |
| INFO       | PQ-007    | Near-perfect accuracy with high ECE      | runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_curve_top1_uncalibrated.csv, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold2/binary_none/calibration_curve_top1_uncalibrated.csv, runs/mrs_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold3/binary_none/calibration_curve_top1_uncalibrated.csv                                                                                                                             | ECE may reflect conservative probabilities or binning noise; prefer confidence-gap and plots. |

### PQ-001 - Low calibration-curve occupancy (WARN)
Condition: Top-1 calibration curve has sparse bins or low sample support; ECE can be unstable.
Measured: `{'min_nonzero_bin_n': 2.0, 'zero_bin_fraction': 0.3, 'n_samples': 94, 'bins': 10}`
Thresholds: `{'min_nonzero_bin_n': 5, 'max_zero_bin_fraction': 0.3, 'min_n_samples': 200}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.final_variant` (Final variant for curve selection)
- `fold1/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy)
- `fold2/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy)
- `fold3/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy)
- `fold1/binary_none/calibration_metadata.json` :: `calibration_metadata.bins` (Configured calibration bins)
Recommendations:
- Interpret ECE qualitatively; inspect reliability plot.
- Use quantile binning if not enabled.
- Avoid gating decisions based on ECE under low occupancy.

### PQ-002 - Underconfidence behavior detected (WARN)
Condition: Predicted confidence is materially lower than observed top-1 accuracy.
Measured: `{'confidence_gap_top1': -0.32413364864335814, 'mean_confidence_top1': 0.6654496846899751, 'accuracy_top1': 0.9895833333333334, 'final_variant': 'uncalibrated'}`
Thresholds: `{'info_gap_lte': -0.1, 'warn_gap_lte': -0.2}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.confidence_gap_top1` (Mean across folds for selected final variant)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.mean_confidence_top1` (Mean confidence context)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1` (Observed top-1 accuracy context)
Recommendations:
- Model is conservative; this is usually safe.
- If probabilities are displayed to clinicians, consider documenting conservatism rather than calibrating.

### PQ-004 - Calibration worsened probability quality (WARN)
Condition: Calibrated outputs regressed beyond configured tolerances relative to uncalibrated outputs.
Measured: `{'delta_brier_recommended_cal_minus_uncal': -0.04524703358391978, 'delta_log_loss_cal_minus_uncal': -0.2820421740774437, 'delta_ece_top1_cal_minus_uncal': -0.11848386261449256, 'delta_ece_ovr_macro_cal_minus_uncal': 0.03339601775573578}`
Thresholds: `{'max_brier_delta': 0.002, 'max_log_loss_delta': 0.01, 'max_ece_top1_delta': 0.02, 'max_ece_ovr_macro_delta': 0.01}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.folds.*.uncalibrated.*` (Uncalibrated metrics source)
- `run.json` :: `artifact_registry.probability_quality.folds.*.calibrated.*` (Calibrated metrics source)
- `run.json` :: `artifact_registry.probability_quality.folds.*.calibration_decision.metrics_compared` (Calibration policy comparison details)
Recommendations:
- Disable calibration or use policy auto mode.
- Retain uncalibrated outputs for final predictions.

### PQ-007 - Near-perfect accuracy with high ECE (INFO)
Condition: High discrimination and high ECE may reflect conservative probabilities or binning artifacts.
Measured: `{'accuracy_top1': 0.9895833333333334, 'ece_top1': 0.33187070561865045}`
Thresholds: `{'accuracy_top1_gte': 0.97, 'ece_top1_gte': 0.15}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1` (Top-1 accuracy source)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.ece_top1` (Top-1 ECE source)
- `fold1/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
- `fold2/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
- `fold3/binary_none/calibration_curve_top1_uncalibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
Recommendations:
- ECE may reflect conservative probabilities or binning noise; prefer confidence-gap and plots.

Next steps decision helper:
- Underconfidence with strong discrimination: keep uncalibrated outputs, document conservatism, and avoid gating on calibration alone.
- Low occupancy detected: increase sample size, reduce bins, enable quantile binning, and interpret ECE qualitatively.
- Calibration worsened quality: disable calibration for final predictions, confirm policy decision, and report both variants.

Promotion gate impact: None (unless user explicitly gates on probability-quality metrics).

## Per-fold Metrics
| metric              |   fold |     value |
|:--------------------|-------:|----------:|
| f1                  |      1 | 0.974425  |
| f1                  |      2 | 1         |
| f1                  |      3 | 1         |
| balanced_accuracy   |      1 | 0.972222  |
| balanced_accuracy   |      2 | 1         |
| balanced_accuracy   |      3 | 1         |
| sensitivity         |      1 | 0.972222  |
| sensitivity         |      2 | 1         |
| sensitivity         |      3 | 1         |
| specificity         |      1 | 0.988095  |
| specificity         |      2 | 1         |
| specificity         |      3 | 1         |
| ppv                 |      1 | 0.979167  |
| ppv                 |      2 | 1         |
| ppv                 |      3 | 1         |
| npv                 |      1 | 0.989583  |
| npv                 |      2 | 1         |
| npv                 |      3 | 1         |
| precision           |      1 | 0.979167  |
| precision           |      2 | 1         |
| precision           |      3 | 1         |
| brier_calibrated    |      1 | 0.0255746 |
| brier_calibrated    |      2 | 0.0216394 |
| brier_calibrated    |      3 | 0.0182996 |
| ece_calibrated      |      1 | 0.220796  |
| ece_calibrated      |      2 | 0.227983  |
| ece_calibrated      |      3 | 0.191382  |
| log_loss_calibrated |      1 | 0.281956  |
| log_loss_calibrated |      2 | 0.2653    |
| log_loss_calibrated |      3 | 0.22397   |

## Calibration Summary
| metric              |     value |
|:--------------------|----------:|
| brier_calibrated    | 0.0218379 |
| ece_calibrated      | 0.213387  |
| log_loss_calibrated | 0.257075  |
_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._

## Probability Calibration Decision
Final variant: `uncalibrated` (`disabled_by_metrics`). Calibration was disabled by policy.
Prediction alignment mismatch rate: `0.0`.

- R4_failed_improvement: brier_recommended delta=+0.003815, required <= -0.002000
- R4_failed_improvement: log_loss delta=+0.034927, allowed <= 0.010000
- R4_failed_improvement: ece_ovr_macro delta=+0.013434, allowed <= 0.010000

| metric            |   uncalibrated |   calibrated |   delta_cal_minus_uncal |
|:------------------|---------------:|-------------:|------------------------:|
| brier_recommended |       0.02176  |    0.0255746 |              0.00381464 |
| log_loss          |       0.247029 |    0.281956  |              0.0349267  |
| ece_top1          |       0.192553 |    0.220796  |              0.0282432  |
| ece_ovr_macro     |       0.101645 |    0.115079  |              0.0134339  |

## Notes
- Calibration method(s): sigmoid
- Calibration bins: 10
- Calibration selection: sigmoid
- Calibration selection rationale: Default to sigmoid calibration.