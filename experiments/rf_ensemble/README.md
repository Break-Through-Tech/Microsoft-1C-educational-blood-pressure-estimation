# Random Forest and ensemble models

Charis Xiong · October 28, 2026

This folder contains the code and results for estimating systolic and diastolic blood pressure from PPG pulse signals. ABP supplies the pressure labels for training; it is not a model input.

## Current results

Adding training recordings and pulse-shape features helped more than the first round of parameter tuning. The latest candidate is a Random Forest with deeper trees, trained on 11,291 windows from 1,308 recordings.

These are **validation MAE** results on the same 857 windows. MAE is the average absolute error in mmHg; lower is better.

| Version | Systolic MAE | Diastolic MAE |
|---|---:|---:|
| Predict the training average | 18.70 | 10.02 |
| Original Random Forest | 16.86 | 8.83 |
| More recordings + pulse features | 16.12 | 8.55 |
| Deeper Random Forest | **15.76** | **8.34** |

The model still misses low and high pressures by a lot. These are development results on a validation set we have checked repeatedly. The newer model needs a fresh final test, and recording IDs do not establish that the patients are different.

## Start here

- [Latest results](RF_REFINEMENT_RESULTS.md): what changed and how the model performed.
- [Latest notebook](rf_refinement.ipynb): saved tables, plots and the training comparison.
- [Training-size comparison](TRAINING_EXPANSION_RESULTS.md): what happened when we added recordings.
- [Preprocessing comparison](PREPROCESSING_COMPARISON_RESULTS.md): normalization, pulse features and filtering.
- [AdaBoost comparison](ADABOOST_RF_RESULTS.md): AdaBoost did not beat the original RF.
- [Earlier test results](ENSEMBLE_FINAL_RESULTS.md): the earlier RF/Extra Trees/boosting comparison. Those test scores are separate from the validation table above.

## Run the latest experiment

Run from this folder. Install `requirements-eda.txt` in a Python environment; the original environment used Python 3.9. Saved notebooks and CSV results can be read without retraining.

```sh
python -m pip install -r requirements-eda.txt
python refine_expanded_rf.py --output-dir outputs/rf_refinement_new_run
python -m unittest test_refine_expanded_rf test_predict_ppg_rf -v
```

This step uses the included feature tables and grouped training predictions. It refits the reference model if its binary is absent. Use a new output directory for each run. Raw MAT files, Python environments and model binaries are not committed.

To rebuild features or expand the training sample, place `Part_1.mat` through `Part_4.mat` in this folder's `data/` directory and follow the corresponding results report. The original features come from `expanded_data_analysis.py`; shared signal helpers are in `signal_preprocessing.py`.

`predict_ppg_rf.py` runs a regenerated model on a 125 Hz PPG recording. See the latest results report for the command and input requirements.
