# Technical Validation Report

## Summary Metrics
| metric                                    |      value |
|:------------------------------------------|-----------:|
| f1_macro                                  |  0.980952  |
| balanced_accuracy                         |  0.983333  |
| recall                                    |  0.983333  |
| specificity                               |  0.997436  |
| ppv                                       |  0.983333  |
| npv                                       |  0.997436  |
| precision                                 |  0.983333  |
| mcc                                       |  0.984496  |
| brier_calibrated                          |  0.0249893 |
| ece_calibrated                            |  0.260046  |
| log_loss_calibrated                       |  0.340141  |
| brier                                     |  0.0249893 |
| brier_recommended                         |  0.0249893 |
| brier_multiclass_sum                      |  0.124946  |
| brier_multiclass_mean                     |  0.0249893 |
| log_loss                                  |  0.340141  |
| ece                                       |  0.260046  |
| ece_top1                                  |  0.260046  |
| ece_ovr_macro                             |  0.10125   |
| mean_confidence_top1                      |  0.72846   |
| accuracy_top1                             |  0.988506  |
| confidence_gap_top1                       | -0.260046  |
| pred_alignment_mismatch_rate              |  0         |
| ece_ovr__G3                               |  0.0975833 |
| ece_ovr__G4                               |  0.111774  |
| ece_ovr__OTHER                            |  0.0822983 |
| ece_ovr__SHH                              |  0.1011    |
| ece_ovr__WNT                              |  0.113494  |
| brier_uncalibrated                        |  0.148738  |
| brier_recommended_uncalibrated            |  0.148738  |
| brier_multiclass_sum_uncalibrated         |  0.74369   |
| brier_multiclass_mean_uncalibrated        |  0.148738  |
| log_loss_uncalibrated                     |  1.47606   |
| ece_uncalibrated                          |  0.771195  |
| ece_top1_uncalibrated                     |  0.771195  |
| ece_ovr_macro_uncalibrated                |  0.30003   |
| mean_confidence_top1_uncalibrated         |  0.228805  |
| accuracy_top1_uncalibrated                |  1         |
| confidence_gap_top1_uncalibrated          | -0.771195  |
| pred_alignment_mismatch_rate_uncalibrated |  0         |
| ece_ovr__G3_uncalibrated                  |  0.307756  |
| ece_ovr__G4_uncalibrated                  |  0.435284  |
| ece_ovr__OTHER_uncalibrated               |  0.244358  |
| ece_ovr__SHH_uncalibrated                 |  0.254776  |
| ece_ovr__WNT_uncalibrated                 |  0.257975  |
| brier_recommended_calibrated              |  0.0249893 |
| brier_multiclass_sum_calibrated           |  0.124946  |
| brier_multiclass_mean_calibrated          |  0.0249893 |
| ece_top1_calibrated                       |  0.260046  |
| ece_ovr_macro_calibrated                  |  0.10125   |
| mean_confidence_top1_calibrated           |  0.72846   |
| accuracy_top1_calibrated                  |  0.988506  |
| confidence_gap_top1_calibrated            | -0.260046  |
| pred_alignment_mismatch_rate_calibrated   |  0         |
| ece_ovr__G3_calibrated                    |  0.0975833 |
| ece_ovr__G4_calibrated                    |  0.111774  |
| ece_ovr__OTHER_calibrated                 |  0.0822983 |
| ece_ovr__SHH_calibrated                   |  0.1011    |
| ece_ovr__WNT_calibrated                   |  0.113494  |

## Probability Quality Checks (ECE/Brier)
These checks are diagnostic and not promotion blockers by default. `ece_top1` evaluates Top-1 confidence calibration; `ece_ovr_macro` evaluates one-vs-rest class probability calibration.

Diagnostic plots generated:
- `probability_quality_plots/prob_quality_reliability_top1.png` - top1 reliability and occupancy thresholds (PQ-001).
- `probability_quality_plots/prob_quality_confidence_gap.png` - Confidence-gap diagnostics with PQ-002/PQ-003 thresholds.
- `probability_quality_plots/prob_quality_calibration_deltas.png` - Calibration deltas against allowed regressions (PQ-004).

| Severity   | Rule ID   | Finding                             | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Recommended next action                                                                       |
|:-----------|:----------|:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| WARN       | PQ-001    | Low calibration-curve occupancy     | runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_curve_top1_calibrated.csv, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold2/binary_none/calibration_curve_top1_calibrated.csv, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold3/binary_none/calibration_curve_top1_calibrated.csv, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_metadata.json | Interpret ECE qualitatively; inspect reliability plot.                                        |
| WARN       | PQ-002    | Underconfidence behavior detected   | runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Model is conservative; this is usually safe.                                                  |
| INFO       | PQ-007    | Near-perfect accuracy with high ECE | runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/run.json, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold1/binary_none/calibration_curve_top1_calibrated.csv, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold2/binary_none/calibration_curve_top1_calibrated.csv, runs/liquid_biopsy_fragmentomic_mb_meta_sklearn_technical/project/runs/technical_validation/technical/fold3/binary_none/calibration_curve_top1_calibrated.csv                                                                                                                                                    | ECE may reflect conservative probabilities or binning noise; prefer confidence-gap and plots. |

### PQ-001 - Low calibration-curve occupancy (WARN)
Condition: top1 calibration curve has sparse bins or low sample support; ECE can be unstable.
Measured: `{'min_nonzero_bin_n': 7.0, 'zero_bin_fraction': 0.0, 'n_samples': 85, 'bins': 10}`
Thresholds: `{'min_nonzero_bin_n': 5, 'max_zero_bin_fraction': 0.3, 'min_n_samples': 200}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.final_variant` (Final variant for curve selection)
- `fold1/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (top1 reliability curve occupancy)
- `fold2/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (top1 reliability curve occupancy)
- `fold3/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (top1 reliability curve occupancy)
- `fold1/binary_none/calibration_metadata.json` :: `calibration_metadata.bins` (Configured calibration bins)
Recommendations:
- Interpret ECE qualitatively; inspect reliability plot.
- Use quantile binning if not enabled.
- Avoid gating decisions based on ECE under low occupancy.

### PQ-002 - Underconfidence behavior detected (WARN)
Condition: Predicted confidence is materially lower than observed top-1 accuracy.
Measured: `{'confidence_gap_top1': -0.2600458381438175, 'mean_confidence_top1': 0.7284599089826194, 'accuracy_top1': 0.9885057471264368, 'final_variant': 'calibrated'}`
Thresholds: `{'info_gap_lte': -0.1, 'warn_gap_lte': -0.2}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.confidence_gap_top1` (Mean across folds for selected final variant)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.mean_confidence_top1` (Mean confidence context)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1` (Observed top-1 accuracy context)
Recommendations:
- Model is conservative; this is usually safe.
- If probabilities are displayed to clinicians, consider documenting conservatism rather than calibrating.

### PQ-007 - Near-perfect accuracy with high ECE (INFO)
Condition: High discrimination and high ECE may reflect conservative probabilities or binning artifacts.
Measured: `{'accuracy_top1': 0.9885057471264368, 'ece_top1': 0.2600458381438174}`
Thresholds: `{'accuracy_top1_gte': 0.97, 'ece_top1_gte': 0.15}`
Evidence:
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.accuracy_top1` (Top-1 accuracy source)
- `run.json` :: `artifact_registry.probability_quality.folds.*.<final_variant>.ece_top1` (Top-1 ECE source)
- `fold1/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
- `fold2/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
- `fold3/binary_none/calibration_curve_top1_calibrated.csv` :: `columns: [bin_id,n,mean_pred,frac_pos]` (Top-1 reliability curve occupancy context)
Recommendations:
- ECE may reflect conservative probabilities or binning noise; prefer confidence-gap and plots.

Next steps decision helper:
- Underconfidence with strong discrimination: keep uncalibrated outputs, document conservatism, and avoid gating on calibration alone.
- Low occupancy detected: increase sample size, reduce bins, enable quantile binning, and interpret ECE qualitatively.

Promotion gate impact: None (unless user explicitly gates on probability-quality metrics).

## Per-fold Metrics
| metric                                    |   fold |      value |
|:------------------------------------------|-------:|-----------:|
| f1                                        |      1 |  0.942857  |
| f1                                        |      2 |  1         |
| f1                                        |      3 |  1         |
| balanced_accuracy                         |      1 |  0.95      |
| balanced_accuracy                         |      2 |  1         |
| balanced_accuracy                         |      3 |  1         |
| sensitivity                               |      1 |  0.95      |
| sensitivity                               |      2 |  1         |
| sensitivity                               |      3 |  1         |
| specificity                               |      1 |  0.992308  |
| specificity                               |      2 |  1         |
| specificity                               |      3 |  1         |
| ppv                                       |      1 |  0.95      |
| ppv                                       |      2 |  1         |
| ppv                                       |      3 |  1         |
| npv                                       |      1 |  0.992308  |
| npv                                       |      2 |  1         |
| npv                                       |      3 |  1         |
| precision                                 |      1 |  0.95      |
| precision                                 |      2 |  1         |
| precision                                 |      3 |  1         |
| mcc                                       |      1 |  0.953488  |
| mcc                                       |      2 |  1         |
| mcc                                       |      3 |  1         |
| brier_calibrated                          |      1 |  0.0279378 |
| brier_calibrated                          |      2 |  0.0259075 |
| brier_calibrated                          |      3 |  0.0211225 |
| ece_calibrated                            |      1 |  0.253431  |
| ece_calibrated                            |      2 |  0.274167  |
| ece_calibrated                            |      3 |  0.25254   |
| log_loss_calibrated                       |      1 |  0.366411  |
| log_loss_calibrated                       |      2 |  0.346042  |
| log_loss_calibrated                       |      3 |  0.30797   |
| brier                                     |      1 |  0.0279378 |
| brier                                     |      2 |  0.0259075 |
| brier                                     |      3 |  0.0211225 |
| brier_recommended                         |      1 |  0.0279378 |
| brier_recommended                         |      2 |  0.0259075 |
| brier_recommended                         |      3 |  0.0211225 |
| brier_multiclass_sum                      |      1 |  0.139689  |
| brier_multiclass_sum                      |      2 |  0.129537  |
| brier_multiclass_sum                      |      3 |  0.105612  |
| brier_multiclass_mean                     |      1 |  0.0279378 |
| brier_multiclass_mean                     |      2 |  0.0259075 |
| brier_multiclass_mean                     |      3 |  0.0211225 |
| log_loss                                  |      1 |  0.366411  |
| log_loss                                  |      2 |  0.346042  |
| log_loss                                  |      3 |  0.30797   |
| ece                                       |      1 |  0.253431  |
| ece                                       |      2 |  0.274167  |
| ece                                       |      3 |  0.25254   |
| ece_top1                                  |      1 |  0.253431  |
| ece_top1                                  |      2 |  0.274167  |
| ece_top1                                  |      3 |  0.25254   |
| ece_ovr_macro                             |      1 |  0.106229  |
| ece_ovr_macro                             |      2 |  0.100002  |
| ece_ovr_macro                             |      3 |  0.097519  |
| mean_confidence_top1                      |      1 |  0.712086  |
| mean_confidence_top1                      |      2 |  0.725833  |
| mean_confidence_top1                      |      3 |  0.74746   |
| accuracy_top1                             |      1 |  0.965517  |
| accuracy_top1                             |      2 |  1         |
| accuracy_top1                             |      3 |  1         |
| confidence_gap_top1                       |      1 | -0.253431  |
| confidence_gap_top1                       |      2 | -0.274167  |
| confidence_gap_top1                       |      3 | -0.25254   |
| pred_alignment_mismatch_rate              |      1 |  0         |
| pred_alignment_mismatch_rate              |      2 |  0         |
| pred_alignment_mismatch_rate              |      3 |  0         |
| ece_ovr__G3                               |      1 |  0.103499  |
| ece_ovr__G3                               |      2 |  0.09544   |
| ece_ovr__G3                               |      3 |  0.0938114 |
| ece_ovr__G4                               |      1 |  0.119331  |
| ece_ovr__G4                               |      2 |  0.104531  |
| ece_ovr__G4                               |      3 |  0.111459  |
| ece_ovr__OTHER                            |      1 |  0.0875638 |
| ece_ovr__OTHER                            |      2 |  0.0792241 |
| ece_ovr__OTHER                            |      3 |  0.080107  |
| ece_ovr__SHH                              |      1 |  0.09861   |
| ece_ovr__SHH                              |      2 |  0.104288  |
| ece_ovr__SHH                              |      3 |  0.100402  |
| ece_ovr__WNT                              |      1 |  0.122142  |
| ece_ovr__WNT                              |      2 |  0.116525  |
| ece_ovr__WNT                              |      3 |  0.101816  |
| brier_uncalibrated                        |      1 |  0.149395  |
| brier_uncalibrated                        |      2 |  0.148378  |
| brier_uncalibrated                        |      3 |  0.14844   |
| brier_recommended_uncalibrated            |      1 |  0.149395  |
| brier_recommended_uncalibrated            |      2 |  0.148378  |
| brier_recommended_uncalibrated            |      3 |  0.14844   |
| brier_multiclass_sum_uncalibrated         |      1 |  0.746977  |
| brier_multiclass_sum_uncalibrated         |      2 |  0.741892  |
| brier_multiclass_sum_uncalibrated         |      3 |  0.742199  |
| brier_multiclass_mean_uncalibrated        |      1 |  0.149395  |
| brier_multiclass_mean_uncalibrated        |      2 |  0.148378  |
| brier_multiclass_mean_uncalibrated        |      3 |  0.14844   |
| log_loss_uncalibrated                     |      1 |  1.4835    |
| log_loss_uncalibrated                     |      2 |  1.47215   |
| log_loss_uncalibrated                     |      3 |  1.47255   |
| ece_uncalibrated                          |      1 |  0.772907  |
| ece_uncalibrated                          |      2 |  0.770234  |
| ece_uncalibrated                          |      3 |  0.770444  |
| ece_top1_uncalibrated                     |      1 |  0.772907  |
| ece_top1_uncalibrated                     |      2 |  0.770234  |
| ece_top1_uncalibrated                     |      3 |  0.770444  |
| ece_ovr_macro_uncalibrated                |      1 |  0.298101  |
| ece_ovr_macro_uncalibrated                |      2 |  0.296662  |
| ece_ovr_macro_uncalibrated                |      3 |  0.305327  |
| mean_confidence_top1_uncalibrated         |      1 |  0.227093  |
| mean_confidence_top1_uncalibrated         |      2 |  0.229766  |
| mean_confidence_top1_uncalibrated         |      3 |  0.229556  |
| accuracy_top1_uncalibrated                |      1 |  1         |
| accuracy_top1_uncalibrated                |      2 |  1         |
| accuracy_top1_uncalibrated                |      3 |  1         |
| confidence_gap_top1_uncalibrated          |      1 | -0.772907  |
| confidence_gap_top1_uncalibrated          |      2 | -0.770234  |
| confidence_gap_top1_uncalibrated          |      3 | -0.770444  |
| pred_alignment_mismatch_rate_uncalibrated |      1 |  0         |
| pred_alignment_mismatch_rate_uncalibrated |      2 |  0         |
| pred_alignment_mismatch_rate_uncalibrated |      3 |  0         |
| ece_ovr__G3_uncalibrated                  |      1 |  0.307336  |
| ece_ovr__G3_uncalibrated                  |      2 |  0.303558  |
| ece_ovr__G3_uncalibrated                  |      3 |  0.312374  |
| ece_ovr__G4_uncalibrated                  |      1 |  0.422539  |
| ece_ovr__G4_uncalibrated                  |      2 |  0.440327  |
| ece_ovr__G4_uncalibrated                  |      3 |  0.442986  |
| ece_ovr__OTHER_uncalibrated               |      1 |  0.255041  |
| ece_ovr__OTHER_uncalibrated               |      2 |  0.223321  |
| ece_ovr__OTHER_uncalibrated               |      3 |  0.254712  |
| ece_ovr__SHH_uncalibrated                 |      1 |  0.248412  |
| ece_ovr__SHH_uncalibrated                 |      2 |  0.257589  |
| ece_ovr__SHH_uncalibrated                 |      3 |  0.258328  |
| ece_ovr__WNT_uncalibrated                 |      1 |  0.257175  |
| ece_ovr__WNT_uncalibrated                 |      2 |  0.258514  |
| ece_ovr__WNT_uncalibrated                 |      3 |  0.258237  |
| brier_recommended_calibrated              |      1 |  0.0279378 |
| brier_recommended_calibrated              |      2 |  0.0259075 |
| brier_recommended_calibrated              |      3 |  0.0211225 |
| brier_multiclass_sum_calibrated           |      1 |  0.139689  |
| brier_multiclass_sum_calibrated           |      2 |  0.129537  |
| brier_multiclass_sum_calibrated           |      3 |  0.105612  |
| brier_multiclass_mean_calibrated          |      1 |  0.0279378 |
| brier_multiclass_mean_calibrated          |      2 |  0.0259075 |
| brier_multiclass_mean_calibrated          |      3 |  0.0211225 |
| ece_top1_calibrated                       |      1 |  0.253431  |
| ece_top1_calibrated                       |      2 |  0.274167  |
| ece_top1_calibrated                       |      3 |  0.25254   |
| ece_ovr_macro_calibrated                  |      1 |  0.106229  |
| ece_ovr_macro_calibrated                  |      2 |  0.100002  |
| ece_ovr_macro_calibrated                  |      3 |  0.097519  |
| mean_confidence_top1_calibrated           |      1 |  0.712086  |
| mean_confidence_top1_calibrated           |      2 |  0.725833  |
| mean_confidence_top1_calibrated           |      3 |  0.74746   |
| accuracy_top1_calibrated                  |      1 |  0.965517  |
| accuracy_top1_calibrated                  |      2 |  1         |
| accuracy_top1_calibrated                  |      3 |  1         |
| confidence_gap_top1_calibrated            |      1 | -0.253431  |
| confidence_gap_top1_calibrated            |      2 | -0.274167  |
| confidence_gap_top1_calibrated            |      3 | -0.25254   |
| pred_alignment_mismatch_rate_calibrated   |      1 |  0         |
| pred_alignment_mismatch_rate_calibrated   |      2 |  0         |
| pred_alignment_mismatch_rate_calibrated   |      3 |  0         |
| ece_ovr__G3_calibrated                    |      1 |  0.103499  |
| ece_ovr__G3_calibrated                    |      2 |  0.09544   |
| ece_ovr__G3_calibrated                    |      3 |  0.0938114 |
| ece_ovr__G4_calibrated                    |      1 |  0.119331  |
| ece_ovr__G4_calibrated                    |      2 |  0.104531  |
| ece_ovr__G4_calibrated                    |      3 |  0.111459  |
| ece_ovr__OTHER_calibrated                 |      1 |  0.0875638 |
| ece_ovr__OTHER_calibrated                 |      2 |  0.0792241 |
| ece_ovr__OTHER_calibrated                 |      3 |  0.080107  |
| ece_ovr__SHH_calibrated                   |      1 |  0.09861   |
| ece_ovr__SHH_calibrated                   |      2 |  0.104288  |
| ece_ovr__SHH_calibrated                   |      3 |  0.100402  |
| ece_ovr__WNT_calibrated                   |      1 |  0.122142  |
| ece_ovr__WNT_calibrated                   |      2 |  0.116525  |
| ece_ovr__WNT_calibrated                   |      3 |  0.101816  |

## Binary Learner Health Report (Meta Under-the-Hood)
Base one-vs-rest learners are internal components feeding the meta model. This section is diagnostic only and is not a promotion gate.

| Class   | AUC mean±std   |   Pred+ rate (mean) |   std(p_pos) (mean) |   n_pos (mean) | Health status   |
|:--------|:---------------|--------------------:|--------------------:|---------------:|:----------------|
| G3      | 0.851±0.068    |               0.283 |              0.3204 |            6.3 | OK              |
| G4      | 0.762±0.096    |               0.518 |              0.2777 |           13   | WARN            |
| OTHER   | 0.814±0.105    |               0.082 |              0.1868 |            2.7 | WARN            |
| SHH     | 0.829±0.063    |               0.071 |              0.2044 |            3.3 | OK              |
| WNT     | 0.957±0.024    |               0.165 |              0.2526 |            3   | OK              |

Warnings:
| Severity   | Rule ID   | Class   | Fold(s)                            | Finding                                                      | Recommended action                                                                             | Evidence paths                                                                                                             |
|:-----------|:----------|:--------|:-----------------------------------|:-------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| WARN       | BL-004    | G4      | fold1:none, fold2:none, fold3:none | High AUC variance across folds: range=0.2250, std=0.0957     | Treat as instability; consider stronger regularization, improved stratification, or more data. | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| WARN       | BL-004    | OTHER   | fold1:none, fold2:none, fold3:none | High AUC variance across folds: range=0.2372, std=0.1051     | Treat as instability; consider stronger regularization, improved stratification, or more data. | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| INFO       | BL-006    | G3      | fold1:none, fold2:none, fold3:none | Low class power detected: mean n_pos=6.33, folds_below_25=3  | Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings.  | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| INFO       | BL-006    | G4      | fold1:none, fold2:none, fold3:none | Low class power detected: mean n_pos=13.00, folds_below_25=3 | Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings.  | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| INFO       | BL-006    | OTHER   | fold1:none, fold2:none, fold3:none | Low class power detected: mean n_pos=2.67, folds_below_25=3  | Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings.  | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| INFO       | BL-006    | SHH     | fold1:none, fold2:none, fold3:none | Low class power detected: mean n_pos=3.33, folds_below_25=3  | Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings.  | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |
| INFO       | BL-006    | WNT     | fold1:none, fold2:none, fold3:none | Low class power detected: mean n_pos=3.00, folds_below_25=3  | Interpret this class's base-learner diagnostics qualitatively; avoid over-weighting warnings.  | `binary_learners_metrics_by_fold.csv`, `binary_learners_metrics_summary.csv`, `run.json#artifact_registry.binary_learners` |

Plots and artifacts:
- `../binary_learners_manifest.json`
- `../binary_learners_metrics_by_fold.csv`
- `../binary_learners_metrics_summary.csv`
- `../binary_learners_warnings.json`
- `../plots/binary_ovr_roc_G3.png`
- `../plots/binary_ovr_roc_G4.png`
- `../plots/binary_ovr_roc_OTHER.png`
- `../plots/binary_ovr_roc_SHH.png`
- `../plots/binary_ovr_roc_WNT.png`
- `../plots/binary_ovr_roc_all_classes.png`
- `../ovo_auc_matrix.csv`
- `../plots/ovo_auc_matrix.png`

Impact on Meta Interpretation: WARN/ERROR findings in `G4, OTHER` indicate potential probability interpretability drift, rare-class instability, and increased reliance on other learners.

## Calibration Summary
| metric                                    |      value |
|:------------------------------------------|-----------:|
| brier_calibrated                          |  0.0249893 |
| ece_calibrated                            |  0.260046  |
| log_loss_calibrated                       |  0.340141  |
| brier                                     |  0.0249893 |
| brier_recommended                         |  0.0249893 |
| brier_multiclass_sum                      |  0.124946  |
| brier_multiclass_mean                     |  0.0249893 |
| log_loss                                  |  0.340141  |
| ece                                       |  0.260046  |
| ece_top1                                  |  0.260046  |
| ece_ovr_macro                             |  0.10125   |
| pred_alignment_mismatch_rate              |  0         |
| brier_uncalibrated                        |  0.148738  |
| brier_recommended_uncalibrated            |  0.148738  |
| brier_multiclass_sum_uncalibrated         |  0.74369   |
| brier_multiclass_mean_uncalibrated        |  0.148738  |
| log_loss_uncalibrated                     |  1.47606   |
| ece_uncalibrated                          |  0.771195  |
| ece_top1_uncalibrated                     |  0.771195  |
| ece_ovr_macro_uncalibrated                |  0.30003   |
| mean_confidence_top1_uncalibrated         |  0.228805  |
| accuracy_top1_uncalibrated                |  1         |
| confidence_gap_top1_uncalibrated          | -0.771195  |
| pred_alignment_mismatch_rate_uncalibrated |  0         |
| ece_ovr__G3_uncalibrated                  |  0.307756  |
| ece_ovr__G4_uncalibrated                  |  0.435284  |
| ece_ovr__OTHER_uncalibrated               |  0.244358  |
| ece_ovr__SHH_uncalibrated                 |  0.254776  |
| ece_ovr__WNT_uncalibrated                 |  0.257975  |
| brier_recommended_calibrated              |  0.0249893 |
| brier_multiclass_sum_calibrated           |  0.124946  |
| brier_multiclass_mean_calibrated          |  0.0249893 |
| ece_top1_calibrated                       |  0.260046  |
| ece_ovr_macro_calibrated                  |  0.10125   |
| mean_confidence_top1_calibrated           |  0.72846   |
| accuracy_top1_calibrated                  |  0.988506  |
| confidence_gap_top1_calibrated            | -0.260046  |
| pred_alignment_mismatch_rate_calibrated   |  0         |
| ece_ovr__G3_calibrated                    |  0.0975833 |
| ece_ovr__G4_calibrated                    |  0.111774  |
| ece_ovr__OTHER_calibrated                 |  0.0822983 |
| ece_ovr__SHH_calibrated                   |  0.1011    |
| ece_ovr__WNT_calibrated                   |  0.113494  |
_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._

## Probability Calibration Decision
Final variant: `calibrated` (`retained`). Calibration was retained by policy.
Prediction alignment mismatch rate: `0.0`.

- R4_passed: calibration improved probability quality within guardrails.

| metric            |   uncalibrated |   calibrated |   delta_cal_minus_uncal |
|:------------------|---------------:|-------------:|------------------------:|
| brier_recommended |       0.149395 |    0.0279378 |               -0.121458 |
| log_loss          |       1.4835   |    0.366411  |               -1.11709  |
| ece_top1          |       0.772907 |    0.253431  |               -0.519476 |
| ece_ovr_macro     |       0.298101 |    0.106229  |               -0.191871 |

## Notes
- Calibration method(s): sigmoid
- Calibration bins: 10
- Calibration selection: sigmoid
- Calibration selection rationale: Default to sigmoid calibration.