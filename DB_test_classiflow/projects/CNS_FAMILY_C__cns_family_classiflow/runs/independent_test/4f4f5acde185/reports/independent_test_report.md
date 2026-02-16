# Independent Test Report

## Metrics
metric,value
n_samples,1104.0
accuracy,0.9067028985507246
balanced_accuracy,0.9289627613822324
f1_macro,0.7842461489248593
f1_weighted,0.9151278323875851
f1_micro,0.9067028985507246
mcc,0.8788406774708771
sensitivity,0.7367635693721154
specificity,0.9961797884849486
ppv,0.6937485260817264
npv,0.9958557653751079
recall,0.7367635693721154
precision,0.6937485260817264
log_loss,1.336400543864335
brier,0.005982963144950681
brier_calibrated,0.005982963144950681
log_loss_calibrated,1.336400543864335
ece,0.08626070264996828
ece_calibrated,0.08626070264996828
ece_top1,0.08626070264996828
ece_binary_pos,
ece_ovr_macro,0.006251104710991933
pred_alignment_mismatch_rate,0.0
calibration_bins,10.0


## Calibration Metrics
metric,value
log_loss,1.336400543864335
brier,0.005982963144950681
brier_calibrated,0.005982963144950681
log_loss_calibrated,1.336400543864335
ece,0.08626070264996828
ece_calibrated,0.08626070264996828
ece_top1,0.08626070264996828
ece_binary_pos,
ece_ovr_macro,0.006251104710991933
pred_alignment_mismatch_rate,0.0
calibration_bins,10.0

_Deprecated aliases: `brier` -> `brier_recommended`, `ece` -> `ece_top1` (temporary compatibility)._

## Notes
- Calibration curve exported as calibration_curve.csv
- Model bundle: model_bundle.zip
- Model trained from scratch on full training data