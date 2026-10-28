# Preprocessing comparison

October 28, 2026. We compared signal normalization, complete-pulse features and a different PPG filter. These changes give small, mixed improvements on the current sample. They do not explain the full difference between our results and another team's results.

## What stayed fixed

All conditions use the same 4,013 training and 857 validation windows, original ABP-derived pressure labels, eligibility rules, recording/overlap groups and three grouped training folds. No test waveforms were processed and no test predictions were generated. The 4,870 development windows cover 567 recordings. Every accessed PPG window matched its saved SHA-256 fingerprint.

RF uses the original 200-tree configuration: depth 10, minimum leaf size 5 and feature fraction 0.8. Boosting uses the earlier fixed configuration: 150 iterations, learning rate 0.05, seven leaves, L2 regularization 10, minimum leaf size 20 and no early stopping. Parameters are fixed across feature conditions. Median imputation and target scaling are fitted inside each training fold.

## Conditions

1. **Original 37:** original PPG/derivative statistics and peak-to-peak summaries.
2. **Normalized 37:** the same calculations on per-window z-scored raw and cleaned PPG. Normalized mean/SD columns are set exactly to 0/1 to avoid treating roundoff as information.
3. **NK pulse shape:** 23 features from complete foot-peak-foot pulses on NeuroKit-cleaned PPG. Features include median and IQR of rise time, fall time, rise/fall ratio, pulse duration, normalized height, upslope, area, rise fraction, widths at 25/50/75% height, and pulse rate. These are candidate measurements, not validated clinical biomarkers.
4. **Original + NK pulse shape:** the original 37 plus those 23 features.
5. **Bandpass pulse shape:** the same 23 pulse features after full-record fourth-order Butterworth filtering at 0.5–8 Hz. Peak locations are held at the NeuroKit detector positions for both pulse-feature variants, so this tests filtering rather than changing filter and detector together. It is not a complete reproduction of Microsoft 1A's pipeline.

Feet are local minima between adjacent peaks; only pulses with both boundaries inside a window are measured. Width is the contiguous region above the specified fraction of a linearly baseline-corrected pulse around its peak. All windows had at least two accepted pulses under both filters, so no rows were dropped. Filtering runs on full recordings before slicing windows. No processing warnings were captured.

## Validation results

All values below are mmHg. Validation has been inspected previously, so these are exploratory development results, not a new final-test score. Mean baseline MAE is 18.70/10.03 mmHg.

| Features / model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE |
|---|---:|---:|---:|---:|
| Original 37 / RF | 16.86 | 8.83 | 22.44 | 12.16 |
| Original 37 / Boosting | 17.12 | 8.89 | 22.53 | 12.24 |
| Normalized 37 / RF | 16.75 | 8.75 | 22.16 | 12.16 |
| Normalized 37 / Boosting | 16.67 | 8.76 | 21.99 | 12.07 |
| NK pulse shape / RF | 17.13 | 8.94 | 22.29 | 12.33 |
| NK pulse shape / Boosting | 16.69 | 8.85 | 21.87 | 12.28 |
| Original + NK pulse shape / RF | 16.54 | 8.81 | 22.08 | 12.17 |
| Original + NK pulse shape / Boosting | 16.65 | 8.91 | 22.01 | 12.39 |
| Bandpass pulse shape / RF | 17.09 | 8.93 | 22.19 | 12.42 |
| Bandpass pulse shape / Boosting | 16.77 | 8.95 | 21.86 | 12.39 |

## What I think

- **Normalization is worth keeping as a candidate feature set.** RF window MAE improves by about 0.11 mmHg SBP and 0.09 DBP. For boosting it improves by 0.45/0.13. However, RF equal-group SBP MAE is slightly worse, so this is not a uniform gain.
- **Complete-pulse features work better as an addition than as a replacement for RF here.** Adding them changes RF window MAE from 16.86/8.83 to 16.54/8.81. Systolic RMSE improves from 22.44 to 22.08, while diastolic RMSE changes little and is slightly worse. The new features help some aspects of the problem but do not solve it.
- **The alternative filter alone shows no consistent advantage.** Compared with NK pulse features, bandpass pulse features give small mixed changes. This experiment does not support switching the entire pipeline to another team's filter.
- **Training-only group-weighted ranking favors normalized features with boosting**, with average target/group OOF MAE 11.79 versus original boosting 11.93. Original+shape RF scores 11.82 versus original RF 11.85. These differences are modest, and selecting among conditions using these scores makes them optimistic estimates of future performance.

`paired_group_gains.csv` contains 2,000 paired group resamples per comparison. Training-OOF intervals are descriptive because fold models share training data. Validation intervals are conditional on fixed fitted candidates and do not account for repeated development decisions or multiple comparisons. A favorable interval for one target is not proof of general improvement.

My recommendation is to carry normalized features and original-plus-pulse features forward into a training-recording learning curve. Expand the number of distinct training recordings before another large parameter search. Keep the current original feature model as the reference. We have not yet tested looser quality gates or alternative pressure labels, so those remain possible contributors. They should be reviewed separately rather than changed to make the score look better.

## Checks and files

Three pulse-measurement tests pass: invariance to positive gain/offset, known synthetic pulse timing/nested widths, and correct handling of incomplete/constant windows. Original RF validation MAE/RMSE exactly reproduce the saved run within floating-point tolerance. The data loader and fold assertions check known identity separation. Both signals and model outputs were checked during the run.

`preprocessing_ablation.py` reproduces the experiment from raw Part_1–4 MAT files and the saved feature/split table. It reads training/validation records only. Use a new OUT directory before rerunning. `test_preprocessing_ablation.py` runs without raw files. `preprocessing_comparison.ipynb` shows the saved results without retraining. The outputs folder holds feature tables, row/label manifest, pulse audit, fold assignments, predictions, metrics, bootstrap differences, training-only ranking and plot. Model binaries stay local and are excluded from upload.

The old test remains a record of the earlier experiment. A future final evaluation needs untouched recordings and their detected overlap relatives excluded from previous development.
