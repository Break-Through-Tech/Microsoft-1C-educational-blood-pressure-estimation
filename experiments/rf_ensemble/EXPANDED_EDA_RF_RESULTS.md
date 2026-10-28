# All-Part EDA and Random Forest results

**Update October 28:** the ensemble comparison and reserved test evaluation are now complete. See [ENSEMBLE_FINAL_RESULTS.md](ENSEMBLE_FINAL_RESULTS.md). Statements below about the reserved test describe the original October 14 run.

October 14, 2026. This continues `data_exploration-2.ipynb` and addresses the coach feedback supplied by the team. The earlier exploration and ResNet pilots are preserved.

## What changed

| Coach request | Implementation and result |
|---|---|
| Sample beyond windows 0–499 | Uniformly sample 200 recording IDs from **each** of Parts 1–4, then up to 12 windows per recording without replacement. Seed 20261014. Actual sample: **7,600 windows from 800 recordings**. |
| Report grouped feature relationships | Report pooled Spearman correlations, recording-median correlations and within-record-centered correlations. Use training/validation only for these feature analyses. |
| Investigate the roughly 120 BPM values | Recompute the first 500 windows under the same full-record detector used for the expanded sample. The old sample covers six recordings and has median PPG/ABP rates 117.19/117.19 BPM; the random all-Part sample has medians 88.24/89.29 BPM before gates. |
| Check for double-counted beats | Compare same-window PPG and ABP **peak** counts, separately report complete ABP intervals, compare median-interval rates, and plot both sets of markers. Record 0/window 0 has 10 PPG and 11 ABP peaks with rates 120.97/122.95 BPM. |
| Filter using `PPG_Quality` and minimum beats | Require mean full-record template quality >=0.8, >=3 PPG peaks and >=3 complete ABP beats; report retention at quality thresholds 0.6/0.7/0.8/0.9 with the other gates fixed. |

Sampling recordings first and limiting their window contributions reduces domination by long recordings. It does not make recordings independent people. No age, sex or patient IDs are available in these MAT files.

## Quality checks and beat review

The screening also checks ABP prominence 5/10/15 mmHg, candidate beat bounds, detector disagreement, nonfinite/constant signals, flat raw extrema and recording-filter boundaries. PPG and ABP peak counts may differ by at most one for phase/window edges; the interval-rate ratio must be between 0.75 and 1.25. These are provisional research rules, not clinical standards.

Before filtering, 434 random windows have a peak-count difference over one and 276 have rate-ratio disagreement. Two have near-2:1 PPG/ABP interval-rate ratios; 190 have near-2:1 ABP/PPG ratios. **A ratio flag does not prove double counting.** The illustrated Part 4 cases show missed/small ABP waves under fixed prominence, and illustrated Part 1 cases show ABP noise/extra candidates. An interval rate derived from a few close noisy peaks can disagree with the whole-window count. Both views matter.

Eight illustrative marker plots were visually inspected and notes saved in `outputs/expanded_eda_oct14/beat_review_notes.csv`. The original record-0 example supports a genuinely faster pulse train rather than obvious PPG doubling. It does not validate all 500 legacy windows. No near-2:1 flagged window was retained under the current gates.

All 806 processed records (800 random plus six legacy-only records) completed without captured processing warnings/errors. This is execution status, not proof of correct signals. Mean quality is template similarity; a high score can occur alongside an unreliable ABP target. Quality 0.8 retains 5,752 windows after all gates/deduplication, versus 5,773 at 0.6 and 5,640 at 0.9. The threshold was not selected using test errors.

## Expanded data and grouping

| Part | Random records | Selected windows | Provisionally retained windows |
|---|---:|---:|---:|
| Part 1 | 200 | 1,778 | 1,335 |
| Part 2 | 200 | 2,048 | 1,603 |
| Part 3 | 200 | 1,833 | 1,336 |
| Part 4 | 200 | 1,941 | 1,478 |
| **Total** | **800** | **7,600** | **5,752** |

The retained data covers 667 recordings in 666 detected overlap groups. One identical selected PPG window was withheld as a duplicate. Full-record hashes and peak-centered one-second exact PPG contexts from all selected recordings were checked against all 12,000 records, finding 139 matched record pairs. Connected detected overlaps stay together. Near-duplicates, overlap chains without a selected anchor, and patient identity are unresolved.

The group split was balanced on Part/record counts only, not pressure targets. Each Part appears in every split:

| Split | Windows | Recordings |
|---|---:|---:|
| Training | 4,013 | 467 |
| Validation | 857 | 100 |
| Reserved test | 882 | 100 |

This is a mixed-Part recording/overlap-group evaluation, not a held-Part-out or proven patient-independent study. Correlations, tuning and importance analysis do not score reserved test predictions.

## What the grouped correlations show

The expanded table contains 37 fixed PPG-only features. For example, VPG skew has SBP Spearman correlation about 0.23 across pooled windows and 0.26 across recording medians, but about -0.01 after centering within recordings. Some associations therefore describe differences **between recordings**, rather than tracking pressure changes inside a recording. These are descriptive comparisons, not causal or independent-patient significance tests.

## Random Forest / ensemble experiment

Inputs include raw/clean PPG statistics, derivative statistics, pulse timing, approximate peak-interval shape summaries and frequency descriptors. ABP values, comparison flags, quality scores and IDs are excluded from the predictor whitelist. Candidate shape summaries are not validated fiducial biomarkers or HRV.

We compared a training-mean baseline, Ridge (alpha 10), Random Forest and Extra Trees. RF used 200 trees and a random 80% of predictors at each split, with four depth/leaf configurations evaluated on three **grouped training folds**. Depth 10/minimum leaf size 5 was selected. Median imputation and target scaling were fitted inside each fold; validation did not select RF hyperparameters. Extra Trees used a predefined configuration and was not separately tuned.

| Model | Validation SBP MAE | Validation DBP MAE | SBP RMSE | DBP RMSE |
|---|---:|---:|---:|---:|
| Training mean | 18.70 | 10.03 | 23.43 | 13.26 |
| Ridge | 17.12 | 8.87 | 22.26 | 12.27 |
| Random Forest | 16.86 | 8.84 | 22.44 | 12.16 |
| Extra Trees | 16.96 | 8.69 | 22.45 | 11.97 |
| RBF SVR | 17.03 | 9.03 | 22.25 | 12.61 |

All errors are mmHg against provisional ABP-derived labels. RF improves window MAE over the mean baseline by about 1.84/1.20 mmHg. An RBF SVR was added using the supplied feature-model references; its four settings were selected with three grouped training folds (C 1/10, epsilon 0.1/0.3). It scored 17.03/9.03 mmHg MAE, close to but slightly behind RF. Extra Trees remained slightly better on DBP. The alternatives are close; this does not establish RF superiority. A histogram-gradient-boosting search was stopped due to its long runtime and has no reported result.

RF recording-average MAE is 16.36/9.17 mmHg. Paired bootstrap resampling of 100 validation overlap groups estimates baseline-to-RF **group-average MAE gains** of 1.99 mmHg SBP (95% percentile interval 0.43–3.75) and 1.28 DBP (0.58–2.02). These intervals describe this validation sample and group assumptions, not clinical or patient-independent accuracy. The corresponding window-weighted and group-weighted numbers should not be mixed.
A second RF tuning pass tested 27 combinations of depth (8/14/unlimited), minimum leaf size (2/5/10) and split feature fraction (0.5/0.8/1.0), still using three grouped training folds. It selected depth 8, leaf size 2 and feature fraction 0.5 (training-CV mean MAE 11.82 mmHg). Validation MAE was 16.91/8.88, essentially unchanged from the original RF at 16.86/8.83. A paired bootstrap of 100 validation overlap groups estimated change in group-macro MAE of +0.12 mmHg SBP improvement (95% interval -0.16 to 0.37) and -0.07 mmHg DBP improvement (-0.20 to 0.07). Both intervals include no change, so the original forest remains the simpler reference. This tuning pass is exploratory and does not alter the reserved test set.

RF errors vary by Part: SBP/DBP MAE is 22.05/13.31 in Part 1, 15.24/8.35 in Part 2, 14.19/6.63 in Part 3 and 16.16/7.11 in Part 4. A validation-only pressure-range analysis confirms shrinkage toward the middle: for SBP below 100, average RF prediction is 129 mmHg (MAE 35.80); for SBP 160 or above, average prediction is 134 mmHg (MAE 39.57). For DBP 90 or above, average prediction is 68 mmHg against a 102 mmHg reference mean (MAE 34.08). RF beats the mean baseline in some extreme bands, but the residual error is still large. This is a modest early result, not evidence of useful clinical accuracy. These are repeated windows from recordings, not independent people. The range table and prediction plots are in `outputs/random_forest_oct14/validation_error_by_pressure_band.csv`, `validation_error_by_pressure_band.png` and `validation_rf_predictions_scatter.png`.

Descriptive validation permutation importance ranks VPG skew, PPG pulse rate and several derivative features highly. Correlated features can share importance; the five-shuffle error bars are not confidence intervals. No feature was removed based on validation importance in this run.

## Where to open the work

- `data_exploration-2.ipynb`: go to **October 14: all-Part random sampling and coach feedback** near the end.
- `random_forest.ipynb`: readable RF setup, grouped training CV, saved comparisons, per-Part/group results, importance, pressure-range error analysis and checkpoint prediction verification.
- `RANDOM_FOREST_RESOURCES.md`: the team's papers/tutorials, implementation choices and resources whose contents could not be retrieved.
- `outputs/expanded_eda_oct14/`: sample IDs, settings, feature audit, processing log, overlap evidence, split manifest, grouped correlations and review plots.
- `outputs/random_forest_oct14/`: fitted forest models, CV assignments/results, validation predictions/metrics, group bootstrap, importance and figures. `outputs/svr_grouped_oct14/`: grouped-CV RBF SVR and same-split validation results. `outputs/rf_tuning_oct14/`: wider grouped-CV RF search and paired validation comparison with the original RF.

## Reproduce

Use the project's `.venv` with `requirements-eda.txt` and keep the four raw MAT files under `data/`:

```sh
.venv/bin/python expanded_data_analysis.py
.venv/bin/python random_forest_pipeline.py --output-dir outputs/random_forest_new_run
.venv/bin/python plot_expanded_analysis.py
.venv/bin/python analyze_rf_validation.py
.venv/bin/python compare_svr_grouped.py
.venv/bin/python tune_random_forest_grouped.py --output-dir outputs/rf_tuning_new_run
.venv/bin/python -m unittest test_expanded_analysis -v
```

The default RF output directory is intentionally not overwritten; use a new run directory. Plotting by default reads the completed October 14 RF run. Generated EDA outputs are regenerated, so retain manual decisions separately and freeze/copy the dataset before changing settings in a model study. Model settings record dataset fingerprints and installed scikit-learn version. Current online docs may describe newer releases.

The earlier ResNet pilot used a different sample and gates. These RF errors must not be used as a direct architecture comparison. Further work should include representative manual review across Parts, filter/label/plateau-rule sensitivity, broader overlap investigation, and a final fixed evaluation protocol. ABP gates are retrospective label-quality checks; a PPG-only deployed predictor cannot require ABP at inference.
