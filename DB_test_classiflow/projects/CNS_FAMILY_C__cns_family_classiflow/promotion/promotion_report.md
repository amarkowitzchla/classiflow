# Promotion Recommendation

**Decision:** PASS

## Promotion Gate
- Template: Legacy Threshold Gates (`legacy_thresholds`)
- Version: 1
- Source: legacy
- Layman explanation: Legacy threshold configuration was used for promotion checks.

### Gate Table
phase,metric,op,threshold,observed_value,passed,scope,aggregation
technical_validation,f1_macro,>=,0.7,0.9677441296527117,True,outer,mean
technical_validation,balanced_accuracy,>=,0.7,0.9706088180029795,True,outer,mean
independent_test,f1_macro,>=,0.7,0.7842461489248593,True,independent,mean
independent_test,balanced_accuracy,>=,0.7,0.9289627613822324,True,independent,mean


## Metrics Used for Promotion Decision
phase,metric,op,threshold,observed_value,passed,scope,aggregation,notes
technical_validation,f1_macro,>=,0.7,0.9677441296527117,True,outer,mean,
technical_validation,balanced_accuracy,>=,0.7,0.9706088180029795,True,outer,mean,
independent_test,f1_macro,>=,0.7,0.7842461489248593,True,independent,mean,
independent_test,balanced_accuracy,>=,0.7,0.9289627613822324,True,independent,mean,


## Supporting Metrics (Reported Only)
phase,metric,value
technical_validation,brier,0.0007875318529646106
technical_validation,brier_recommended,0.0007875318529646106
technical_validation,brier_multiclass_sum,0.022838423735973706
technical_validation,brier_multiclass_mean,0.0007875318529646106
technical_validation,log_loss,0.11999584973938554
technical_validation,ece,0.010547222367001416
technical_validation,ece_top1,0.010547222367001416
technical_validation,ece_ovr_macro,0.0004372748467389288
technical_validation,pred_alignment_mismatch_rate,0.0
technical_validation,brier_uncalibrated,0.0007875318529646106
technical_validation,log_loss_uncalibrated,0.11999584973938554
technical_validation,ece_uncalibrated,0.010547222367001416
independent_test,accuracy,0.9067028985507246
independent_test,log_loss,1.336400543864335
independent_test,brier,0.005982963144950681
independent_test,ece,0.08626070264996828
independent_test,ece_top1,0.08626070264996828
independent_test,ece_binary_pos,
independent_test,ece_ovr_macro,0.006251104710991933
independent_test,pred_alignment_mismatch_rate,0.0
independent_test,calibration_bins,10.0


## Calibration Summary
- Selected method: unavailable

## Gate Evaluation
phase,passed,reasons
technical_validation,True,
independent_test,True,
