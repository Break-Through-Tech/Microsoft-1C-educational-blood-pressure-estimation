# SVR (RBF kernel): PPG only, Sophia

## Data
- **Part 1 only**, 5-second windows (625 samples at 125 Hz).
- Labels: SBP = median of ABP peaks, DBP = median of ABP valleys.
- Dropped implausible windows: SBP outside 60–220, DBP outside 30–130, or SBP − DBP < 10.
- **Up to 10 randomly chosen windows per recording** (seed 42), so long recordings don't dominate. This leaves **24,186 windows from 2,990 recordings**.
- 80/20 train/test split **grouped by recording** (seed 42): 19,329 train windows (2,392 recordings) and 4,857 test windows (598 recordings).
- 44 PPG features, all from the PPG signal: heart rate, pulse amplitude/width/area, fiducial ratios and timings (dicrotic notch, diastolic peak), and summary statistics of PPG, VPG and APG.

## Model and settings
- Pipeline: median imputation → StandardScaler → SVR (RBF). SBP and DBP use separate models.
- Tuning: `RandomizedSearchCV` with 20 iterations and 5-fold `GroupKFold` by recording, on a 4,000-window sample of the training set. The best settings were then refit on the full training set.
  - SBP: C = 291.5, epsilon = 5, gamma = 0.01
  - DBP: C = 6.34, epsilon = 1, gamma = 0.05
- **Weighted variant:** the same settings, but rare BP values get higher sample weights (inverse square root of the count in their 10-mmHg bin, capped at 10×), to push back against predictions being squashed toward the average.

## Results (test set)

| Target | Model | MAE | RMSE | R² | Bias | Slope |
|---|---|---|---|---|---|---|
| SBP | Mean baseline | 19.91 | 24.15 | 0.00 | −1.20 | 0.00 |
| SBP | **SVR** | **13.12** | 17.68 | 0.46 | +0.44 | 0.49 |
| SBP | SVR (weighted) | 13.27 | 18.03 | 0.44 | +1.72 | 0.55 |
| DBP | Mean baseline | 9.02 | 11.54 | 0.00 | +0.14 | 0.00 |
| DBP | **SVR** | **6.80** | 9.80 | 0.28 | −1.00 | 0.30 |
| DBP | SVR (weighted) | 7.76 | 10.52 | 0.17 | +2.64 | 0.42 |

Errors are in mmHg. Full metrics are in `metrics.csv` and per-range errors in `metrics_by_bp_range.csv`. Neither model meets the AAMI standard (|bias| ≤ 5 and SD ≤ 8 mmHg).

**Plots:** `predicted_vs_actual.png`, `bland_altman.png`, `error_by_bp_range.png` and `feature_importance.png` (permutation importance). BP ranges are based on the actual values and are scored separately for SBP and DBP. DBP has no "Elevated" category.

## Key takeaways
- **SVR beats the mean baseline:** MAE drops by 34% for SBP (19.9 → 13.1) and 25% for DBP (9.0 → 6.8).
- **Predictions are squashed toward the average.** The slopes of predicted vs. actual are 0.49 (SBP) and 0.30 (DBP), so the model overestimates low BP and underestimates high BP. Errors are largest at the extremes, for example SBP < 90 (MAE 24.5) and DBP ≥ 90 (MAE 27.2).
- **Weighting rare values helps at the extremes but hurts overall.** It cuts DBP Stage 2 error from 27.2 to 19.5 and raises the slopes, but overall MAE gets worse. This is a trade-off, not a free improvement.
- **Pulse-shape features matter most.** The top features are the kurtosis and skew of the first derivative (VPG), followed by APG statistics and PPG amplitude.
- Educational prototype only, not clinically validated. Recording IDs are not confirmed to be unique patients.
