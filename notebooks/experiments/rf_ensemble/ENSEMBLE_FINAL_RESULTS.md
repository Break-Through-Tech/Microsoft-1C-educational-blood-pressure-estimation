# Completed Random Forest and ensemble comparison

October 28, 2026. The RF/ensemble experiment is complete for the current frozen feature dataset. The test results are now available; the October 14 files still describe the earlier validation run.

## Final test results

Errors below are in mmHg on 882 test windows from 100 recordings / 100 detected overlap groups across Parts 1–4. Training used 4,013 windows; validation used 857. These are recording/overlap-group splits, not confirmed independent patients.

| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE |
|---|---:|---:|---:|---:|
| Training mean | 19.56 | 8.21 | 23.04 | 10.59 |
| Random Forest | 19.00 | 7.72 | 23.73 | 9.71 |
| Extra Trees | 18.60 | 7.69 | 23.34 | 9.66 |
| Histogram boosting | 19.34 | 7.61 | 23.74 | 9.61 |
| Equal-weight ensemble | 18.91 | 7.61 | 23.48 | 9.56 |

Extra Trees has the lowest observed SBP MAE. The equal-weight ensemble has the lowest observed DBP MAE, virtually tied with boosting (7.608 versus 7.609 before rounding). The differences are small, so there is no clear winner. RF improves MAE over the mean baseline by only 0.56 mmHg SBP and 0.49 mmHg DBP; its SBP RMSE is worse than baseline. When we resampled the test recording groups, the 95% intervals for each model’s improvement over baseline included zero. The models therefore show only a small improvement over predicting the training average.

## What we ran

- Reused the frozen October 14 RF and Extra Trees training checkpoints after verifying dataset fingerprints. Their validation metrics reproduce the saved results.
- Completed histogram gradient boosting: four configurations on three grouped training folds, with separate target regressors, training-fold target scaling and native missing-value handling. Selected seven leaves and L2 regularization 10. Fixed 150 iterations, learning rate 0.05, minimum leaf size 20; early stopping disabled to avoid an internal ungrouped validation split.
- Added a predefined arithmetic average of RF, Extra Trees and boosting, with one-third weights. No validation/test weight fitting.
- Wrote the protocol before generating test predictions. Reported every frozen candidate on validation and test, with per-Part metrics and 2,000 paired overlap-group bootstrap resamples. No refitting or tuning followed test scoring.
- Existing five data/split/quality tests passed. Runtime checks confirmed original validation reproduction, finite predictions, and dataset fingerprints; existing loader checks prevent known identity overlap across splits.

## Files and reproduction

`finish_ensemble_comparison.py` reproduces the run. Set OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2; run with `.venv/bin/python`. Its output directory must not already exist, so archive/copy the current results and change OUT for any new run. The current test has now been examined and must not become a tuning set for later experiments.

`outputs/ensemble_final_oct28/` contains the protocol, selected boosting settings, fitted boosting checkpoint, CV scores, validation/test predictions, metrics, per-Part breakdowns and bootstrap intervals. RF and Extra Trees checkpoints remain in `outputs/random_forest_oct14/`. To reproduce the equal-weight predictor, average those two checkpoints' predictions and the new boosting checkpoint's predictions on the same ordered feature whitelist in protocol.json. All three return mmHg.

## What the results mean

The comparison is finished, but the errors are still too large to call the model accurate. Labels and signal-quality gates remain provisional, and patient identities and undetected overlaps remain unresolved. Further gains likely require reviewing labels/features and pressure-range coverage. Any later model development needs a new untouched evaluation dataset or a rigorously specified new evaluation design; these test results cannot be used to tune and still count as an untouched final test.
