# Latest Random Forest results

October 28, 2026

A deeper Random Forest improved the expanded model's validation MAE from **16.12/8.55 to 15.76/8.34 mmHg** for systolic/diastolic pressure. Both RMSE values also improved. This is the candidate to carry forward, although large errors remain at low and high pressures.

## What changed

The training data stayed at 11,291 windows from 1,308 recordings, with 60 PPG features. The labels, quality rules and validation windows stayed the same.

We compared four changes with the expanded reference:

- Allow deeper trees in the shared systolic/diastolic forest.
- Fit a separate forest for each pressure.
- Give each recording-overlap group equal total weight in the reference forest.
- Combine separate forests with equal-group weights.

The selected model uses **200 trees, unlimited depth, minimum leaf size 5 and 80% of features considered at each split**. Its only change from the expanded reference is removing the depth-10 limit.

## How it was chosen

We split training data into three folds, keeping detected overlap groups together. Each prediction below comes from a model that did not train on that group. Errors are averaged within each group first, so a recording with more windows does not dominate the score. Missing-value filling and pressure scaling are fitted on each training fold only.

A replacement had to improve the average of the two group errors by at least 0.05 mmHg without worsening either pressure. This rule was written before fitting. The deeper shared forest had the lowest combined score, though the leading alternatives were close.

| Model | Group systolic MAE | Group diastolic MAE |
|---|---:|---:|
| Expanded reference RF | 15.31 | 7.56 |
| Deeper shared RF | **14.94** | **7.44** |
| Separate-target RF | 14.91 | 7.47 |
| Equal-group reference RF | 15.38 | 7.54 |
| Equal-group separate RF | 14.97 | 7.41 |

## Validation check

After saving the training-only decision, we scored the reference and selected model on the same 857 validation windows. The old test was not scored again. All errors are mmHg.

| Model | Systolic MAE | Diastolic MAE | Systolic RMSE | Diastolic RMSE |
|---|---:|---:|---:|---:|
| Expanded reference RF | 16.12 | 8.55 | 21.49 | 11.65 |
| Deeper shared RF | **15.76** | **8.34** | **21.23** | **11.44** |

MAE is the average absolute error. RMSE gives large misses more weight. The mean-prediction baseline on this validation set has MAE of 18.70/10.02.

When validation groups were resampled, the 95% interval for the systolic gain included zero; the diastolic interval was positive. These checks do not account for all the times we have inspected validation. The new model still needs an independent final evaluation.

## What still needs work

Predictions remain too close to the middle. In training-fold predictions, the lowest systolic quarter was overpredicted by about 19 mmHg, and the highest quarter was underpredicted by about 23 mmHg. Better average error has not solved those misses.

Next, review representative training signals and pressure labels before another large parameter search. The labels are still provisional, and recording IDs do not prove patient independence.

## Code and checks

- `refine_expanded_rf.py`: reruns the comparison. Use a new `--output-dir`.
- `rf_refinement.ipynb`: saved results and pressure-range plots.
- `outputs/rf_refinement_oct28/`: settings, selection decision, predictions, metrics and checks.
- `predict_ppg_rf.py`: runs a saved model on a one-dimensional NumPy PPG recording at 125 Hz.

Twelve checks passed, covering sampling, group separation, pulse measurements, weighting and prediction inputs. Raw-PPG prediction also matched the saved-feature prediction for a training example. Model binaries stay local and can be regenerated.

For a regenerated model, predict a five-second window with at least one second of signal context on each side:

```sh
python predict_ppg_rf.py --ppg-npy recording.npy --start-sample 625 --model outputs/rf_refinement_new_run/selected_rf.joblib
```

The script checks PPG quality and pulse counts. ABP-based reference checks cannot be performed from PPG alone.
