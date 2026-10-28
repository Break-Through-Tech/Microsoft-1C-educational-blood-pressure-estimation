# Training-recording learning curve

October 28, 2026. This experiment adds training recordings while keeping the original validation windows fixed. All results are exploratory development results; the old test was not scored.

## Training size

| Training set | Recordings | Windows |
|---|---:|---:|
| Original training | 467 | 4013 |
| Half expansion | 886 | 7585 |
| Full expansion | 1308 | 11291 |

Each new source recording was sampled without looking at pressure values. The same quality gates and pressure-label method were used as before. The expansion samples 250 source recordings per Part, up to 12 windows per recording. The half/full subsets are nested, with seeded new-group order. Duplicate and shifted-overlap detection was rerun before training. Training records connected to held-out identities were excluded. The preparation summary records exclusions and retention.

## Validation MAE

All entries are mmHg. The same fixed RF and boosting settings are used at every size and with every feature set. Lower is better.

| Training set | Features | Model | SBP MAE | DBP MAE |
|---|---|---|---:|---:|
| Original training | Original | RF | 16.86 | 8.83 |
| Original training | Original | Boosting | 17.12 | 8.89 |
| Original training | Normalized | RF | 16.75 | 8.75 |
| Original training | Normalized | Boosting | 16.67 | 8.76 |
| Original training | Original + pulse | RF | 16.54 | 8.81 |
| Original training | Original + pulse | Boosting | 16.68 | 8.95 |
| Original training | Constant | mean | 18.70 | 10.03 |
| Original training | Constant | median | 18.93 | 10.21 |
| Half expansion | Original | RF | 16.63 | 8.61 |
| Half expansion | Original | Boosting | 16.93 | 8.76 |
| Half expansion | Normalized | RF | 16.45 | 8.76 |
| Half expansion | Normalized | Boosting | 16.41 | 8.72 |
| Half expansion | Original + pulse | RF | 16.44 | 8.68 |
| Half expansion | Original + pulse | Boosting | 16.69 | 8.81 |
| Half expansion | Constant | mean | 18.68 | 10.02 |
| Half expansion | Constant | median | 18.92 | 10.15 |
| Full expansion | Original | RF | 16.34 | 8.61 |
| Full expansion | Original | Boosting | 16.85 | 8.59 |
| Full expansion | Normalized | RF | 16.10 | 8.68 |
| Full expansion | Normalized | Boosting | 16.24 | 8.69 |
| Full expansion | Original + pulse | RF | 16.12 | 8.55 |
| Full expansion | Original + pulse | Boosting | 16.64 | 8.65 |
| Full expansion | Constant | mean | 18.70 | 10.02 |
| Full expansion | Constant | median | 18.96 | 10.17 |

## What we learned

The expanded RF with original-plus-pulse features is the most promising candidate. It has the lowest average target/group OOF error on the fixed original training windows. Its validation MAE is 16.12 SBP / 8.55 DBP mmHg, compared with 16.86/8.83 for the original-size RF with original features. Validation RMSE improves from 22.44/12.16 to 21.49/11.65. The full-size mean baseline remains about 18.70/10.02 MAE, so a shifted constant prediction does not explain the gains.

Normalized RF has a nearly identical systolic validation MAE of 16.10, but a higher diastolic MAE of 8.68. I would carry forward the added-pulse RF because its fixed-population training OOF comparison is stronger, rather than pick a model just for its lowest validation systolic number.

More training recordings gave a steadier improvement than the previous parameter changes. The gain is still modest, and most paired validation intervals for expansion include zero. This is a better development candidate, not evidence of final accuracy. Keep the original model and all earlier test results as references. The next useful investigation is a representative training-only label/signal review before deciding whether more data, label refinements or further features are worthwhile.

## How to read this experiment

The graph uses a fixed validation population. Adding training recordings can help some feature/model combinations and hurt others; assess both targets, RMSE and equal-group MAE, not just the lowest cell. Constant baselines are refitted at each size so changes in training pressure distributions remain visible.

`fixed_original_oof_metrics.csv` also compares each training size on the same original 4,013 training windows using their out-of-fold predictions. Fold membership is held fixed across sizes, so these original evaluation groups never enter the corresponding training fold. This provides a training-only comparison on a fixed population, alongside the exploratory validation comparison.

Three fixed group folds are defined on the complete training pool and restricted to each nested subset. Every model/feature condition at a given size shares those folds. OOF predictions cover the corresponding training population, so those errors are not comparisons on the same evaluation population across sizes. Validation is the fixed-population comparison.

`paired_expansion_gains.csv` reports equal-group MAE change against the original-size model with the same features. Positive is improvement. The 95% intervals use 2,000 paired group resamples and are conditional on these fixed candidates; they do not account for repeatedly inspecting validation or multiple comparisons.

## Scope and limitations

No test labels, test feature table or test predictions were used for model fitting or selection. Raw source signals, including held-out source PPG, were scanned only to check overlap. This is a research leakage audit, not a test performance evaluation. Patient identity and undetected near-overlaps remain unknown. Quality gates and labels are provisional. The held-out validation set has been inspected repeatedly, so improvements need confirmation on a new untouched evaluation set before being treated as final.

## Files and checks

`expand_training_recordings.py` samples and prepares the training additions. `training_recording_learning_curve.py` generates new features and fits the fixed models. `summarize_training_learning_curve.py` creates the paired comparisons, report and chart from saved outputs. Output directories must be new for a fresh run. Raw data and model binaries are excluded from upload.

The saved outputs include sampling IDs, gate/overlap audits, feature tables, source hashes, folds, nested group membership, training counts, predictions and MAE/RMSE/bias/equal-group metrics. Data-loading checks, per-window source hashes, finite-prediction checks and group separation assertions run before or during fitting. The original-size RF conditions can be compared with earlier saved validation values to check reproduction.
