# Random Forest with ECG — Charis

## Result to report

**Random Forest using PPG and ECG reached validation MAE of 15.03 mmHg systolic and 7.63 mmHg diastolic. About 77.8% of diastolic predictions were within 10 mmHg.**

These are five-second-window results on 857 validation windows from 100 recordings. Training used 11,291 windows from 1,308 recordings. The model uses 60 PPG features and 31 ECG/cross-signal features. ABP supplies the reference labels only.

## Did ECG help?

Both versions use the same windows, pressure labels, folds and RF settings.

| Inputs | Systolic MAE | Diastolic MAE | Systolic RMSE | Diastolic RMSE | DBP within 10 |
|---|---:|---:|---:|---:|---:|
| PPG only | 15.76 | 8.34 | 21.23 | 11.44 | 69.5% |
| PPG + ECG | **15.03** | **7.63** | **20.43** | **10.92** | **77.8%** |
| Predict training average | 18.70 | 10.02 | 23.44 | 13.17 | 58.8% |

All pressure errors are mmHg; lower is better. “Within 10” means absolute error <=10 mmHg. It is not an overall accuracy score. The saved metrics also include percentages within 5 and 15, signed error, R² and equal-group MAE.

## What the RF does

The model averages 200 decision trees. Trees have no maximum-depth limit, require at least five samples per leaf and consider 80% of features at each split. Missing values and target scaling are fitted using training data only. Model checks use three folds that keep detected recording-overlap groups together.

ECG adds waveform statistics, normalized shape, R-peak timing and signal-power features. We also tried an ECG-to-PPG peak-delay summary. That delay is often missing under the provisional matching bounds and is not a validated pulse transit time measurement. No ABP value, recording ID or pressure-derived timing is a predictor.

## What this means

Adding ECG helped this RF on the matched validation set. It does not establish clinical accuracy or independent-patient performance. The pressure labels are provisional, and low/high-pressure errors still need work.

Validation has been inspected repeatedly during development. The old test was not scored again. A fresh final evaluation is still needed before presenting these as final test results.

## Files and reproduction

- `rf_ecg.ipynb`: saved tables and explanations.
- `train_rf_ecg.py`: refits both RF versions from the included frozen tables. No raw MAT files are needed for this step.
- `outputs/rf_ecg_nov1/`: features, labels, folds, predictions, metrics and ECG checks.
- `predict_ppg_ecg.py`: runs a regenerated two-output model on synchronized 125 Hz PPG/ECG recordings.
- `prepare_ecg_comparison.py` and `audit_ecg_groups.py`: optional ECG feature rebuilding and exact-repeat checks; these need the raw MAT files.

Run from this folder:

```sh
python -m pip install -r requirements-eda.txt
python train_rf_ecg.py --output-dir outputs/rf_ecg_new_run
python -m unittest test_standard_bp_metrics test_ecg_features test_ppg_ecg_prediction -v
```

The output directory must be new. Generated model binaries stay local and are excluded from Git. To use a regenerated ECG RF:

```sh
python predict_ppg_ecg.py --ppg-npy ppg.npy --ecg-npy ecg.npy --start-sample 625 --model outputs/rf_ecg_new_run/ppg_and_ecg_random_forest.joblib
```

Inputs must be synchronized one-dimensional signals at 125 Hz, with at least one second of context on each side of the five-second window. PPG checks are applied; ABP reference checks are unavailable at inference.
