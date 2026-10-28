# Random Forest and ensemble experiment

Charis Xiong · October 28, 2026

This folder contains the PPG feature experiment: Random Forest, Extra Trees, histogram gradient boosting, and an equal-weight average of those three models. The comparison is complete for the current dataset.

Start with [the results](ENSEMBLE_FINAL_RESULTS.md). The [RF notebook](random_forest.ipynb) explains the earlier validation experiments and ends with the final test results. Saved tables and figures are under `outputs/`; they can be read without running the models.

On 882 test windows, RF achieved SBP/DBP MAE of 19.00/7.72 mmHg. Extra Trees had the lowest SBP MAE (18.60); the combined ensemble had the lowest DBP MAE (7.61). Improvements over the training-mean baseline were small, and the group-bootstrap intervals include no improvement. The labels remain provisional and the split does not establish patient independence.

## Run the code

Run commands from this folder. Install `requirements-eda.txt` in a Python environment (the original run used Python 3.9). Raw MAT files and generated model binaries are not committed.

1. Put `Part_1.mat` through `Part_4.mat` in `data/` here, or link this folder's `data` to the repository's root `data` folder.
2. `python expanded_data_analysis.py` regenerates the feature dataset. This is a substantial signal-processing run and overwrites generated EDA tables, so preserve the committed outputs before rerunning.
3. `python random_forest_pipeline.py --output-dir outputs/random_forest_new_run` trains RF, Extra Trees and baselines. The output directory must be new.
4. In `finish_ensemble_comparison.py`, point `OLD` to that forest run and set `OUT` to a new directory. Run `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python finish_ensemble_comparison.py` to train boosting and report the fixed ensemble.
5. `python -m unittest test_expanded_analysis -v` checks sampling, group separation and quality gates; these checks need the raw MAT files.

The notebook includes saved outputs. Cells that load `.joblib` checkpoints require a model rerun and an updated `RUN` path. The training scripts check dataset fingerprints, so changed data preparation cannot silently reuse the original checkpoints.

The test set has now been evaluated. Do not use these results to tune models and then describe the same test as untouched. The earlier October 14 notes describe validation-only work; the October 28 report contains the final comparison.

## Shared data preparation

`expanded_data_analysis.py`, `signal_preprocessing.py`, and `plot_expanded_analysis.py` prepare and check the PPG features and ABP-derived labels used by the forest models. They do not train a neural network. The saved feature table is included so the modeling input and splits can be inspected. The wider tuning script and its outputs are included because they tune Random Forest. SVR and ResNet experiments are excluded.

## AdaBoost and RF follow-up

See [ADABOOST_RF_RESULTS.md](ADABOOST_RF_RESULTS.md) and [adaboost_rf.ipynb](adaboost_rf.ipynb). AdaBoost did not beat the original RF; separate-target RF gave only a small window-level change, with slightly worse equal-group MAE. These are validation-only experiments; the earlier test was not rescored. `compare_adaboost_rf.py` reproduces the new comparison after the original model checkpoints are regenerated.
