# AdaBoost and Random Forest refinement results

October 28, 2026. We tried AdaBoost and two RF changes on the same 37 PPG features. All settings were chosen using three overlap-grouped training folds. The table below is exploratory validation on the same 857 windows from 100 recordings, not a new test result. The earlier test split was not scored again.

| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE |
|---|---:|---:|---:|---:|
| Training mean | 18.70 | 10.03 | 23.43 | 13.26 |
| Training median | 18.93 | 10.21 | 23.83 | 13.92 |
| Original RF | 16.86 | 8.83 | 22.44 | 12.16 |
| Original Extra Trees | 16.96 | 8.69 | 22.45 | 11.97 |
| AdaBoost | 17.72 | 9.11 | 22.91 | 12.49 |
| Regularized RF | 16.97 | 8.88 | 22.37 | 12.17 |
| Separate-target RF | 16.77 | 8.80 | 22.25 | 12.18 |

All errors are in mmHg. Lower is better. These validation numbers must not be compared directly with the earlier 882-window test table.

## What we learned

AdaBoost beats the training-mean baseline, but performs worse than the original RF on MAE and RMSE for both pressures. It is a useful completed comparison, not the strongest next model on these features.

The separate-target RF makes a small MAE improvement over the original: about 0.09 mmHg for SBP and 0.04 for DBP. SBP RMSE also improves by 0.19, while DBP RMSE is slightly worse. Both target-wise 95% paired group-bootstrap intervals for its MAE gain over original RF include zero. When each recording/overlap group counts equally, it is slightly worse than original RF: group-average MAE increases by 0.14 mmHg SBP and 0.07 DBP. This is too small and inconsistent to claim a meaningful improvement.

The regularized RF reduces the training/validation gap, but its validation MAE is slightly worse than original RF. A smaller gap alone is not enough. The separate-target RF still has a large gap: training MAE is 7.35/3.58 versus validation 16.77/8.80, so generalization remains the main issue.

## Settings and checks

- AdaBoost: eight settings tested; selected depth 4, learning rate 0.05 and 50 estimators. Each weak tree requires at least 10 samples per leaf; loss is linear. Separate regressors fit SBP and DBP.
- Regularized RF: four settings; selected depth 8 and leaf size 10, with 200 trees, feature fraction 0.5 and bootstrap sample fraction 0.6.
- Separate-target RF: four settings; selected unlimited depth and leaf size 5, with 200 trees per target, feature fraction 0.5 and bootstrap sample fraction 0.7. Both targets use the same selected hyperparameters but fit separate trees.
- Selection minimizes equal-weight SBP/DBP window MAE in mmHg. RMSE is also saved. Training-fold median imputation and target scaling are refitted inside each fold.
- Five existing sampling/group/quality tests passed. Dataset fingerprints and fold separation checks passed. Original RF, Extra Trees and mean-baseline validation MAE/RMSE reproduced their saved values.
- CV best-setting scores are selection scores, not unbiased estimates. Validation has been inspected repeatedly, and bootstrap intervals do not account for the whole model-selection process.

## Recommendation

Keep the original RF as the reference and the separate-target version as an exploratory alternative. AdaBoost does not justify replacing RF here. The next experiment should add explicit foot-peak-foot pulse timing/shape features and expand the number of training recordings, using grouped training folds. Compare feature sets on identical rows before attributing gains to features. Review labels rather than deleting hard examples just to lower errors. Do not use the old test to tune these changes.

## Open or rerun

`adaboost_rf.ipynb` explains the experiment and shows saved results. `compare_adaboost_rf.py` contains the training code; its CLI accepts `--data-dir`, `--original-dir` and `--output-dir`. Use a new output directory. Inputs are the saved feature dataset and original model checkpoints. Generated binaries are kept locally and excluded from the upload.

`outputs/adaboost_rf_oct28/` contains the protocol, grouped-CV results, selected settings, train/validation metrics and predictions, per-Part errors, group-bootstrap results and comparison plot. This experiment does not change the earlier final-test results.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
