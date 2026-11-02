# Random Forest with ECG — Charis

## Best RF result

Random Forest using **PPG + ECG** features, evaluated on **857 five-second validation windows from 100 recordings**.

| Metric | Systolic | Diastolic |
|---|---:|---:|
| MAE | **15.03 mmHg** | **7.63 mmHg** |
| RMSE | **20.43 mmHg** | **10.92 mmHg** |
| Predictions within 10 mmHg | **49.0%** | **77.8%** |

MAE is the average absolute error; lower is better. “Within 10” is the percentage with absolute error <=10 mmHg, not classification accuracy.

Training used 11,291 windows from 1,308 recordings, with 60 PPG features and 31 ECG features. ABP supplied reference labels only. The RF uses 200 trees, unlimited depth, minimum leaf size 5 and 80% of features considered per split.

These are validation results from development. The labels remain provisional, extreme-pressure errors remain large, and a fresh final evaluation is still needed. Recording IDs do not prove patient independence.

## Files

- [Results and methods](RF_ECG_RESULTS.md)
- [Notebook with saved results](rf_ecg.ipynb)
- `train_rf_ecg.py`: reproduce this RF comparison from the included feature tables.
- `predict_ppg_ecg.py`: use a regenerated model with synchronized 125 Hz PPG/ECG signals.
- `outputs/rf_ecg_nov1/`: data manifest, features, folds, predictions and metrics.

Run from this folder with the project environment:

```sh
python -m pip install -r requirements-eda.txt
python train_rf_ecg.py --output-dir outputs/rf_ecg_new_run
```

Use a new output directory. Raw recordings and model binaries are not committed. Earlier experiment files remain for reproducibility; their scores are not the current RF result.
