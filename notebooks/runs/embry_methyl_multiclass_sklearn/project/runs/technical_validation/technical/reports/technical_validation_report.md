# Technical Validation Report

## Summary Metrics
| metric                                    |       value |
|:------------------------------------------|------------:|
| f1_macro                                  |  0.954026   |
| balanced_accuracy                         |  0.95245    |
| recall                                    |  0.95245    |
| specificity                               |  0.995286   |
| ppv                                       |  0.959763   |
| npv                                       |  0.995401   |
| precision                                 |  0.959763   |
| mcc                                       |  0.949211   |
| brier                                     |  0.00618949 |
| brier_recommended                         |  0.00618949 |
| brier_multiclass_sum                      |  0.0680844  |
| brier_multiclass_mean                     |  0.00618949 |
| log_loss                                  |  0.149825   |
| ece                                       |  0.0622491  |
| ece_top1                                  |  0.0622491  |
| ece_ovr_macro                             |  0.010311   |
| mean_confidence_top1                      |  0.901697   |
| accuracy_top1                             |  0.962103   |
| confidence_gap_top1                       | -0.0604059  |
| pred_alignment_mismatch_rate              |  0          |
| ece_ovr__ATRT_MYC                         |  0.00929752 |
| ece_ovr__ATRT_SHH                         |  0.014761   |
| ece_ovr__ATRT_TYR                         |  0.00718685 |
| ece_ovr__CNS_BCOR_ITD                     |  0.00486861 |
| ece_ovr__CNS_NB_FOXR2                     |  0.00496017 |
| ece_ovr__ETMR_C19MC                       |  0.00257369 |
| ece_ovr__MB_G3                            |  0.0211279  |
| ece_ovr__MB_G4                            |  0.0233815  |
| ece_ovr__MB_SHH_CHL_AD                    |  0.00854743 |
| ece_ovr__MB_SHH_INF                       |  0.0111026  |
| ece_ovr__MB_WNT                           |  0.00561424 |
| brier_uncalibrated                        |  0.00618949 |
| brier_recommended_uncalibrated            |  0.00618949 |
| brier_multiclass_sum_uncalibrated         |  0.0680844  |
| brier_multiclass_mean_uncalibrated        |  0.00618949 |
| log_loss_uncalibrated                     |  0.149825   |
| ece_uncalibrated                          |  0.0622491  |
| ece_top1_uncalibrated                     |  0.0622491  |
| ece_ovr_macro_uncalibrated                |  0.010311   |
| mean_confidence_top1_uncalibrated         |  0.901697   |
| accuracy_top1_uncalibrated                |  0.962103   |
| confidence_gap_top1_uncalibrated          | -0.0604059  |
| pred_alignment_mismatch_rate_uncalibrated |  0          |
| ece_ovr__ATRT_MYC_uncalibrated            |  0.00929752 |
| ece_ovr__ATRT_SHH_uncalibrated            |  0.014761   |
| ece_ovr__ATRT_TYR_uncalibrated            |  0.00718685 |
| ece_ovr__CNS_BCOR_ITD_uncalibrated        |  0.00486861 |
| ece_ovr__CNS_NB_FOXR2_uncalibrated        |  0.00496017 |
| ece_ovr__ETMR_C19MC_uncalibrated          |  0.00257369 |
| ece_ovr__MB_G3_uncalibrated               |  0.0211279  |
| ece_ovr__MB_G4_uncalibrated               |  0.0233815  |
| ece_ovr__MB_SHH_CHL_AD_uncalibrated       |  0.00854743 |
| ece_ovr__MB_SHH_INF_uncalibrated          |  0.0111026  |
| ece_ovr__MB_WNT_uncalibrated              |  0.00561424 |

## Probability Quality Checks (ECE/Brier)
These checks are diagnostic and not promotion blockers by default. `ece_top1` evaluates Top-1 confidence calibration; `ece_ovr_macro` evaluates one-vs-rest class probability calibration.

Diagnostic plots generated:
- `probability_quality_plots/prob_quality_reliability_top1.png` - top1 reliability and occupancy thresholds (PQ-001).
- `probability_quality_plots/prob_quality_confidence_gap.png` - Confidence-gap diagnostics with PQ-002/PQ-003 thresholds.

_No probability-quality checks were triggered or required artifacts were unavailable._

Promotion gate impact: None (unless user explicitly gates on probability-quality metrics).

## Per-fold Metrics
| metric                                    |   fold |        value |
|:------------------------------------------|-------:|-------------:|
| f1                                        |      1 |  0.962484    |
| f1                                        |      2 |  0.973864    |
| f1                                        |      3 |  0.9338      |
| f1                                        |      4 |  0.893867    |
| f1                                        |      5 |  0.978477    |
| f1                                        |      6 |  0.978048    |
| f1                                        |      7 |  0.938129    |
| f1                                        |      8 |  0.903936    |
| f1                                        |      9 |  0.981626    |
| f1                                        |     10 |  0.979725    |
| f1                                        |     11 |  0.973623    |
| f1                                        |     12 |  0.950731    |
| balanced_accuracy                         |      1 |  0.958027    |
| balanced_accuracy                         |      2 |  0.971241    |
| balanced_accuracy                         |      3 |  0.927257    |
| balanced_accuracy                         |      4 |  0.887798    |
| balanced_accuracy                         |      5 |  0.982105    |
| balanced_accuracy                         |      6 |  0.979064    |
| balanced_accuracy                         |      7 |  0.932639    |
| balanced_accuracy                         |      8 |  0.90168     |
| balanced_accuracy                         |      9 |  0.982872    |
| balanced_accuracy                         |     10 |  0.983926    |
| balanced_accuracy                         |     11 |  0.973623    |
| balanced_accuracy                         |     12 |  0.949164    |
| sensitivity                               |      1 |  0.958027    |
| sensitivity                               |      2 |  0.971241    |
| sensitivity                               |      3 |  0.927257    |
| sensitivity                               |      4 |  0.887798    |
| sensitivity                               |      5 |  0.982105    |
| sensitivity                               |      6 |  0.979064    |
| sensitivity                               |      7 |  0.932639    |
| sensitivity                               |      8 |  0.90168     |
| sensitivity                               |      9 |  0.982872    |
| sensitivity                               |     10 |  0.983926    |
| sensitivity                               |     11 |  0.973623    |
| sensitivity                               |     12 |  0.949164    |
| specificity                               |      1 |  0.996315    |
| specificity                               |      2 |  0.996727    |
| specificity                               |      3 |  0.994283    |
| specificity                               |      4 |  0.990839    |
| specificity                               |      5 |  0.996363    |
| specificity                               |      6 |  0.996231    |
| specificity                               |      7 |  0.993181    |
| specificity                               |      8 |  0.990738    |
| specificity                               |      9 |  0.998532    |
| specificity                               |     10 |  0.998026    |
| specificity                               |     11 |  0.996848    |
| specificity                               |     12 |  0.995354    |
| ppv                                       |      1 |  0.969089    |
| ppv                                       |      2 |  0.977955    |
| ppv                                       |      3 |  0.948467    |
| ppv                                       |      4 |  0.905213    |
| ppv                                       |      5 |  0.976256    |
| ppv                                       |      6 |  0.977834    |
| ppv                                       |      7 |  0.952773    |
| ppv                                       |      8 |  0.914209    |
| ppv                                       |      9 |  0.980919    |
| ppv                                       |     10 |  0.978114    |
| ppv                                       |     11 |  0.973623    |
| ppv                                       |     12 |  0.962704    |
| npv                                       |      1 |  0.996575    |
| npv                                       |      2 |  0.997034    |
| npv                                       |      3 |  0.994694    |
| npv                                       |      4 |  0.991245    |
| npv                                       |      5 |  0.996154    |
| npv                                       |      6 |  0.996253    |
| npv                                       |      7 |  0.993373    |
| npv                                       |      8 |  0.990859    |
| npv                                       |      9 |  0.998464    |
| npv                                       |     10 |  0.997887    |
| npv                                       |     11 |  0.996848    |
| npv                                       |     12 |  0.995427    |
| precision                                 |      1 |  0.969089    |
| precision                                 |      2 |  0.977955    |
| precision                                 |      3 |  0.948467    |
| precision                                 |      4 |  0.905213    |
| precision                                 |      5 |  0.976256    |
| precision                                 |      6 |  0.977834    |
| precision                                 |      7 |  0.952773    |
| precision                                 |      8 |  0.914209    |
| precision                                 |      9 |  0.980919    |
| precision                                 |     10 |  0.978114    |
| precision                                 |     11 |  0.973623    |
| precision                                 |     12 |  0.962704    |
| mcc                                       |      1 |  0.961011    |
| mcc                                       |      2 |  0.966753    |
| mcc                                       |      3 |  0.938889    |
| mcc                                       |      4 |  0.899403    |
| mcc                                       |      5 |  0.960986    |
| mcc                                       |      6 |  0.960683    |
| mcc                                       |      7 |  0.927098    |
| mcc                                       |      8 |  0.898886    |
| mcc                                       |      9 |  0.983185    |
| mcc                                       |     10 |  0.977743    |
| mcc                                       |     11 |  0.966238    |
| mcc                                       |     12 |  0.949653    |
| brier                                     |      1 |  0.00476962  |
| brier                                     |      2 |  0.0106733   |
| brier                                     |      3 |  0.00312559  |
| brier_recommended                         |      1 |  0.00476962  |
| brier_recommended                         |      2 |  0.0106733   |
| brier_recommended                         |      3 |  0.00312559  |
| brier_multiclass_sum                      |      1 |  0.0524658   |
| brier_multiclass_sum                      |      2 |  0.117406    |
| brier_multiclass_sum                      |      3 |  0.0343814   |
| brier_multiclass_mean                     |      1 |  0.00476962  |
| brier_multiclass_mean                     |      2 |  0.0106733   |
| brier_multiclass_mean                     |      3 |  0.00312559  |
| log_loss                                  |      1 |  0.0999929   |
| log_loss                                  |      2 |  0.270031    |
| log_loss                                  |      3 |  0.0794496   |
| ece                                       |      1 |  0.012491    |
| ece                                       |      2 |  0.138515    |
| ece                                       |      3 |  0.0357414   |
| ece_top1                                  |      1 |  0.012491    |
| ece_top1                                  |      2 |  0.138515    |
| ece_top1                                  |      3 |  0.0357414   |
| ece_ovr_macro                             |      1 |  0.00578539  |
| ece_ovr_macro                             |      2 |  0.0216652   |
| ece_ovr_macro                             |      3 |  0.00348259  |
| mean_confidence_top1                      |      1 |  0.958556    |
| mean_confidence_top1                      |      2 |  0.797129    |
| mean_confidence_top1                      |      3 |  0.949407    |
| accuracy_top1                             |      1 |  0.965517    |
| accuracy_top1                             |      2 |  0.935644    |
| accuracy_top1                             |      3 |  0.985149    |
| confidence_gap_top1                       |      1 | -0.00696129  |
| confidence_gap_top1                       |      2 | -0.138515    |
| confidence_gap_top1                       |      3 | -0.0357414   |
| pred_alignment_mismatch_rate              |      1 |  0           |
| pred_alignment_mismatch_rate              |      2 |  0           |
| pred_alignment_mismatch_rate              |      3 |  0           |
| ece_ovr__ATRT_MYC                         |      1 |  0.00327209  |
| ece_ovr__ATRT_MYC                         |      2 |  0.0226238   |
| ece_ovr__ATRT_MYC                         |      3 |  0.0019967   |
| ece_ovr__ATRT_SHH                         |      1 |  0.00637984  |
| ece_ovr__ATRT_SHH                         |      2 |  0.0281683   |
| ece_ovr__ATRT_SHH                         |      3 |  0.00973488  |
| ece_ovr__ATRT_TYR                         |      1 |  0.00486021  |
| ece_ovr__ATRT_TYR                         |      2 |  0.0130693   |
| ece_ovr__ATRT_TYR                         |      3 |  0.00363105  |
| ece_ovr__CNS_BCOR_ITD                     |      1 |  0.00457164  |
| ece_ovr__CNS_BCOR_ITD                     |      2 |  0.00811881  |
| ece_ovr__CNS_BCOR_ITD                     |      3 |  0.00191537  |
| ece_ovr__CNS_NB_FOXR2                     |      1 |  0.000982495 |
| ece_ovr__CNS_NB_FOXR2                     |      2 |  0.0130693   |
| ece_ovr__CNS_NB_FOXR2                     |      3 |  0.0008287   |
| ece_ovr__ETMR_C19MC                       |      1 |  0.000554347 |
| ece_ovr__ETMR_C19MC                       |      2 |  0.00618812  |
| ece_ovr__ETMR_C19MC                       |      3 |  0.000978597 |
| ece_ovr__MB_G3                            |      1 |  0.0150889   |
| ece_ovr__MB_G3                            |      2 |  0.0435149   |
| ece_ovr__MB_G3                            |      3 |  0.00478002  |
| ece_ovr__MB_G4                            |      1 |  0.0145178   |
| ece_ovr__MB_G4                            |      2 |  0.0474257   |
| ece_ovr__MB_G4                            |      3 |  0.00820111  |
| ece_ovr__MB_SHH_CHL_AD                    |      1 |  0.00618395  |
| ece_ovr__MB_SHH_CHL_AD                    |      2 |  0.0175743   |
| ece_ovr__MB_SHH_CHL_AD                    |      3 |  0.00188409  |
| ece_ovr__MB_SHH_INF                       |      1 |  0.00599377  |
| ece_ovr__MB_SHH_INF                       |      2 |  0.0237129   |
| ece_ovr__MB_SHH_INF                       |      3 |  0.00360102  |
| ece_ovr__MB_WNT                           |      1 |  0.00123432  |
| ece_ovr__MB_WNT                           |      2 |  0.0148515   |
| ece_ovr__MB_WNT                           |      3 |  0.000756919 |
| brier_uncalibrated                        |      1 |  0.00476962  |
| brier_uncalibrated                        |      2 |  0.0106733   |
| brier_uncalibrated                        |      3 |  0.00312559  |
| brier_recommended_uncalibrated            |      1 |  0.00476962  |
| brier_recommended_uncalibrated            |      2 |  0.0106733   |
| brier_recommended_uncalibrated            |      3 |  0.00312559  |
| brier_multiclass_sum_uncalibrated         |      1 |  0.0524658   |
| brier_multiclass_sum_uncalibrated         |      2 |  0.117406    |
| brier_multiclass_sum_uncalibrated         |      3 |  0.0343814   |
| brier_multiclass_mean_uncalibrated        |      1 |  0.00476962  |
| brier_multiclass_mean_uncalibrated        |      2 |  0.0106733   |
| brier_multiclass_mean_uncalibrated        |      3 |  0.00312559  |
| log_loss_uncalibrated                     |      1 |  0.0999929   |
| log_loss_uncalibrated                     |      2 |  0.270031    |
| log_loss_uncalibrated                     |      3 |  0.0794496   |
| ece_uncalibrated                          |      1 |  0.012491    |
| ece_uncalibrated                          |      2 |  0.138515    |
| ece_uncalibrated                          |      3 |  0.0357414   |
| ece_top1_uncalibrated                     |      1 |  0.012491    |
| ece_top1_uncalibrated                     |      2 |  0.138515    |
| ece_top1_uncalibrated                     |      3 |  0.0357414   |
| ece_ovr_macro_uncalibrated                |      1 |  0.00578539  |
| ece_ovr_macro_uncalibrated                |      2 |  0.0216652   |
| ece_ovr_macro_uncalibrated                |      3 |  0.00348259  |
| mean_confidence_top1_uncalibrated         |      1 |  0.958556    |
| mean_confidence_top1_uncalibrated         |      2 |  0.797129    |
| mean_confidence_top1_uncalibrated         |      3 |  0.949407    |
| accuracy_top1_uncalibrated                |      1 |  0.965517    |
| accuracy_top1_uncalibrated                |      2 |  0.935644    |
| accuracy_top1_uncalibrated                |      3 |  0.985149    |
| confidence_gap_top1_uncalibrated          |      1 | -0.00696129  |
| confidence_gap_top1_uncalibrated          |      2 | -0.138515    |
| confidence_gap_top1_uncalibrated          |      3 | -0.0357414   |
| pred_alignment_mismatch_rate_uncalibrated |      1 |  0           |
| pred_alignment_mismatch_rate_uncalibrated |      2 |  0           |
| pred_alignment_mismatch_rate_uncalibrated |      3 |  0           |
| ece_ovr__ATRT_MYC_uncalibrated            |      1 |  0.00327209  |
| ece_ovr__ATRT_MYC_uncalibrated            |      2 |  0.0226238   |
| ece_ovr__ATRT_MYC_uncalibrated            |      3 |  0.0019967   |
| ece_ovr__ATRT_SHH_uncalibrated            |      1 |  0.00637984  |
| ece_ovr__ATRT_SHH_uncalibrated            |      2 |  0.0281683   |
| ece_ovr__ATRT_SHH_uncalibrated            |      3 |  0.00973488  |
| ece_ovr__ATRT_TYR_uncalibrated            |      1 |  0.00486021  |
| ece_ovr__ATRT_TYR_uncalibrated            |      2 |  0.0130693   |
| ece_ovr__ATRT_TYR_uncalibrated            |      3 |  0.00363105  |
| ece_ovr__CNS_BCOR_ITD_uncalibrated        |      1 |  0.00457164  |
| ece_ovr__CNS_BCOR_ITD_uncalibrated        |      2 |  0.00811881  |
| ece_ovr__CNS_BCOR_ITD_uncalibrated        |      3 |  0.00191537  |
| ece_ovr__CNS_NB_FOXR2_uncalibrated        |      1 |  0.000982495 |
| ece_ovr__CNS_NB_FOXR2_uncalibrated        |      2 |  0.0130693   |
| ece_ovr__CNS_NB_FOXR2_uncalibrated        |      3 |  0.0008287   |
| ece_ovr__ETMR_C19MC_uncalibrated          |      1 |  0.000554347 |
| ece_ovr__ETMR_C19MC_uncalibrated          |      2 |  0.00618812  |
| ece_ovr__ETMR_C19MC_uncalibrated          |      3 |  0.000978597 |
| ece_ovr__MB_G3_uncalibrated               |      1 |  0.0150889   |
| ece_ovr__MB_G3_uncalibrated               |      2 |  0.0435149   |
| ece_ovr__MB_G3_uncalibrated               |      3 |  0.00478002  |
| ece_ovr__MB_G4_uncalibrated               |      1 |  0.0145178   |
| ece_ovr__MB_G4_uncalibrated               |      2 |  0.0474257   |
| ece_ovr__MB_G4_uncalibrated               |      3 |  0.00820111  |
| ece_ovr__MB_SHH_CHL_AD_uncalibrated       |      1 |  0.00618395  |
| ece_ovr__MB_SHH_CHL_AD_uncalibrated       |      2 |  0.0175743   |
| ece_ovr__MB_SHH_CHL_AD_uncalibrated       |      3 |  0.00188409  |
| ece_ovr__MB_SHH_INF_uncalibrated          |      1 |  0.00599377  |
| ece_ovr__MB_SHH_INF_uncalibrated          |      2 |  0.0237129   |
| ece_ovr__MB_SHH_INF_uncalibrated          |      3 |  0.00360102  |
| ece_ovr__MB_WNT_uncalibrated              |      1 |  0.00123432  |
| ece_ovr__MB_WNT_uncalibrated              |      2 |  0.0148515   |
| ece_ovr__MB_WNT_uncalibrated              |      3 |  0.000756919 |

## Calibration Summary
| metric                                    |       value |
|:------------------------------------------|------------:|
| brier                                     |  0.00618949 |
| brier_recommended                         |  0.00618949 |
| brier_multiclass_sum                      |  0.0680844  |
| brier_multiclass_mean                     |  0.00618949 |
| log_loss                                  |  0.149825   |
| ece                                       |  0.0622491  |
| ece_top1                                  |  0.0622491  |
| ece_ovr_macro                             |  0.010311   |
| pred_alignment_mismatch_rate              |  0          |
| brier_uncalibrated                        |  0.00618949 |
| brier_recommended_uncalibrated            |  0.00618949 |
| brier_multiclass_sum_uncalibrated         |  0.0680844  |
| brier_multiclass_mean_uncalibrated        |  0.00618949 |
| log_loss_uncalibrated                     |  0.149825   |
| ece_uncalibrated                          |  0.0622491  |
| ece_top1_uncalibrated                     |  0.0622491  |
| ece_ovr_macro_uncalibrated                |  0.010311   |
| mean_confidence_top1_uncalibrated         |  0.901697   |
| accuracy_top1_uncalibrated                |  0.962103   |
| confidence_gap_top1_uncalibrated          | -0.0604059  |
| pred_alignment_mismatch_rate_uncalibrated |  0          |
| ece_ovr__ATRT_MYC_uncalibrated            |  0.00929752 |
| ece_ovr__ATRT_SHH_uncalibrated            |  0.014761   |
| ece_ovr__ATRT_TYR_uncalibrated            |  0.00718685 |
| ece_ovr__CNS_BCOR_ITD_uncalibrated        |  0.00486861 |
| ece_ovr__CNS_NB_FOXR2_uncalibrated        |  0.00496017 |
| ece_ovr__ETMR_C19MC_uncalibrated          |  0.00257369 |
| ece_ovr__MB_G3_uncalibrated               |  0.0211279  |
| ece_ovr__MB_G4_uncalibrated               |  0.0233815  |
| ece_ovr__MB_SHH_CHL_AD_uncalibrated       |  0.00854743 |
| ece_ovr__MB_SHH_INF_uncalibrated          |  0.0111026  |
| ece_ovr__MB_WNT_uncalibrated              |  0.00561424 |
_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._