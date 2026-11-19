# Charis — Random Forest and AdaBoost results

Both models estimate SBP and DBP from **PPG + ECG**, using the same 68 input features, labels, windows and recording split.

| Model | SBP MAE (mmHg) | DBP MAE (mmHg) | SBP RMSE (mmHg) | DBP RMSE (mmHg) | Results |
|---|---:|---:|---:|---:|---|
| Random Forest | 9.91 | 5.34 | 14.36 | 8.22 | [RandomForest](RandomForest/) |
| AdaBoost | 13.65 | 7.59 | 17.68 | 10.43 | [AdaBoost](AdaBoost/) |

These scores use the same 19,480 test-subset windows from 450 recordings. Lower errors are better. Random Forest had the lowest observed errors of these two fixed configurations.

## Files

Each model folder contains:

- `metrics.csv`: SBP/DBP validation and test-subset metrics, plus the training-mean baseline.
- `predicted_vs_actual.png`: predictions against reference labels, with the perfect-prediction diagonal.
- `bland_altman.png`: prediction error against average BP, including bias and bias ±1.96 SD.
- `error_by_bp_range.png`: MAE and window counts for Low, Normal, Elevated, Stage 1 and Stage 2.
- `summary.md`: model settings, preprocessing, results and limitations.
- `features.md`: exact model input columns.

[Shared feature definitions](features.md) explain all 68 inputs. No code, raw signals or model binaries are included.

## Shared data and split

Part 1 only; all eligible nonoverlapping five-second windows (625 samples at 125 Hz). Labels are medians of detected raw ABP peaks/valleys, with minimum detection spacing 50 samples. Retain SBP 60–220, DBP 30–130 and SBP−DBP ≥10 mmHg; exclude invalid labels or nonfinite PPG/ABP.

The supplied NumPy default_rng(42) permutation splits unique eligible recording IDs 70/15/15: training 91,377 windows / 2,094 recordings; validation 20,012 / 448; test 19,480 / 450. All windows from a recording stay together. ECG missingness does not change these rows. Imputation and target scaling use training statistics only. Settings were fixed before scoring this follow-up.

## Interpretation

Adding ECG lowered observed errors for both models compared with their matched PPG-only references (RF: 11.82/6.47; AdaBoost: 15.10/8.41 mmHg SBP/DBP MAE). The two submitted folders contain the PPG + ECG runs only.

Previously explored recordings and evaluated test-subset data are reused. These are exploratory standardized comparisons, not an untouched final test or clinical validation. Recording IDs do not establish unique patients, and cross-record duplicates are not removed by this requested split. Different filtering or recording lists across teammates can still produce different evaluation populations. ECG-to-PPG delay features have substantial missingness, as documented in features.md.
