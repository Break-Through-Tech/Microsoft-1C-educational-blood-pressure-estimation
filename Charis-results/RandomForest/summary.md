# Random Forest — PPG + ECG — Charis

Part 1 only, all 130,869 retained five-second windows (625 samples at 125 Hz). Uses the exact same saved rows, labels and seed-42 70/15/15 recording split shared by both submitted models: 91,377 training / 20,012 validation / 19,480 test windows. SBP/DBP are medians of detected ABP peaks/valleys; bounds 60–220 / 30–130 mmHg and pulse pressure ≥10. No ECG-based exclusions. See features.md for 68 inputs.

Settings: {"n_estimators": 200, "max_depth": null, "min_samples_leaf": 5, "max_features": 0.8, "random_state": 42, "n_jobs": 4, "verbose": 1}. Separate SBP/DBP AdaBoost regressors; joint-output RF. Training-only median imputation and target standardization. Settings fixed from earlier development; no tuning against this run's validation or test subset.

Test-subset MAE: **9.91 SBP / 5.34 DBP mmHg**. RMSE: 14.36 / 8.22. Compare with the mean baseline in metrics.csv. Figures use test windows; the per-category MAE values and counts are shown in the error-by-range PNG.

Scatter includes a perfect-prediction diagonal. Bland–Altman uses predicted-minus-actual against their average, with bias ±1.96 sample SD. BP bins use actual labels, Low taking precedence: SBP<90 or DBP<60; otherwise Stage 2 if SBP≥140 or DBP≥90; Stage 1 if SBP≥130 or DBP≥80; Elevated if SBP≥120; otherwise Normal. Descriptive bins, not diagnoses.

Previously explored recordings and evaluated test-subset data are reused: these are exploratory standardized comparisons, not a fresh final test. Recording IDs are not confirmed unique patients; cross-record duplicates are not removed by this requested split. The prototype is not clinically validated. Earlier multi-Part results stay separate. Only results and documentation are included in this folder.

## Matched comparison takeaway
Adding ECG reduced RF test-subset MAE from 11.82/6.47 to 9.91/5.34 mmHg (SBP/DBP). This RF has the lowest observed errors among the four fixed model/input configurations. ECG-to-PPG delay median/IQR are missing in 62.7% of all retained windows and are imputed from training medians.
