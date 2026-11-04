# Data and model sources

The signal sample comes from:

Kachuee, M., Kiani, M., Mohammadzade, H., & Shabany, M. (2015). *Cuff-Less Blood Pressure Estimation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B602

Dataset page: https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation

UCI lists the dataset under **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

## Included sample

`assets/signals.npz` repackages eight selected PPG and ECG recordings from Parts 1–4, with five-second ABP excerpts for display. Original signal values are retained; the sample is a subset of the dataset, not a new collection. Full PPG/ECG recording context is used for filtering and peak detection. Two records per Part and one eligible validation window per record were selected with NumPy RNG seed 20261104, without consulting errors. `assets/examples.csv` identifies each recording, window and start sample.

The project's median complete-beat ABP labels are included in `examples.csv`; they are derived labels rather than a window-wide maximum/minimum. ABP is not passed to prediction. The sample has already been used for development validation and must not be described as a fresh test set.

## Model

`assets/rf_ecg.joblib` is the project's two-output RF trained on 11,291 windows from 1,308 recordings. It is a compressed copy of the November 1 PPG + ECG model, with unchanged predictions. The preprocessing code is copied from the same experiment's feature and inference functions. The 91 predictors contain PPG and ECG features only. Missing-value imputation and target scaling were fitted on training data.

`assets/validation_metrics.csv` contains the original matched RF validation metrics. Reproducing training is covered by the parent experiment's scripts and results, rather than performed during this demo.
