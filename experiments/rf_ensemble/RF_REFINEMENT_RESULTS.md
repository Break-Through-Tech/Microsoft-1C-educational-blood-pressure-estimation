# Expanded RF refinement

October 28, 2026. This follow-up tests whether more flexible trees, separate target forests or equal-group weighting improve the expanded RF. The feature set stays fixed at the original 37 plus 23 complete-pulse features. The training pool stays at 1,308 recordings / 11,291 windows.

## Training-only selection

All candidates use the same three overlap-group folds as the full learning curve. Training-fold median imputation and target scaling remain inside each fit. No new feature selection, quality filtering, pressure bounds or label changes were made.

- **Deeper joint RF:** change maximum depth from 10 to unlimited; keep 200 trees, minimum leaf size 5, feature fraction 0.8 and seed 20261014.
- **Separate-target RF:** the same deeper configuration, with separate SBP and DBP forests.
- **Group-balanced reference RF:** retain the original depth-10 model and give each overlap group equal total training weight.
- **Group-balanced separate RF:** separate deeper forests with equal-group weights.

Weights are inverse window counts per overlap group, normalized to mean one and recomputed inside each training fold. No record ID, quality score or ABP diagnostic enters the predictors.

A replacement must reduce mean target/group OOF MAE by at least 0.05 mmHg and must not worsen either target's group OOF MAE. This rule was written before fitting. Validation was scored only for the reference and the training-selected replacement, after the selection JSON was saved. The old test was not accessed.

| Model | Group OOF SBP MAE | Group OOF DBP MAE |
|---|---:|---:|
| Deeper joint RF | 14.94 | 7.44 |
| Expanded reference RF | 15.31 | 7.56 |
| Group-balanced reference RF | 15.38 | 7.54 |
| Group-balanced separate RF | 14.97 | 7.41 |
| Separate-target RF | 14.91 | 7.47 |

Training-selected candidate: **Deeper joint RF**. These OOF selection scores are optimistic after choosing among candidates; they are not independent final estimates.

## Exploratory validation

Same 857 validation windows; all values are mmHg.

| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE |
|---|---:|---:|---:|---:|
| Deeper joint RF | 15.76 | 8.34 | 21.23 | 11.44 |
| Expanded reference RF | 16.12 | 8.55 | 21.49 | 11.65 |

`validation_paired_group_gains.csv` reports 2,000 paired group resamples for the selected candidate versus the expanded reference. Positive is improvement. The intervals are conditional on these models and do not account for repeatedly inspected validation or all earlier model-development choices.

## What I would keep

Keep **Deeper joint RF** as the next development candidate. Its validation MAE is 15.76/8.34, versus 16.12/8.55 for the expanded reference. Both RMSE values also improved in this run. The paired equal-group intervals are saved separately; they do not account for the full repeated development process.

The harder problem remains pressure-range generalization. The selected model's lowest/highest training SBP quartiles still have biases about +19.12/-22.79 mmHg. A better average error has not fixed these large misses. I would review representative training signals and labels next instead of expanding into another broad parameter search.

## Pressure-range behavior

The reference predicts toward the middle: its training-OOF lowest/highest SBP quartiles have signed bias about +19.91/-24.16 mmHg. The graph and `training_pressure_quartile_errors.csv` show whether the selected refinement reduces those biases. These quartiles are defined only from training labels and are not selection features. Better average MAE does not mean extreme-pressure errors are solved.

## Running predictions

`predict_ppg_rf.py` loads a trusted local model and a one-dimensional NumPy PPG recording sampled at 125 Hz. It filters the full recording, extracts the same original and pulse features, and predicts one five-second window. Supply at least one second of signal context on each side. It checks finite/nonconstant PPG, pulse count, template quality and flat raw extrema. ABP-based label-quality checks cannot be performed from PPG alone.

Example, after regenerating the model binary:

```sh
python predict_ppg_rf.py --ppg-npy recording.npy --start-sample 625 --model outputs/rf_refinement_oct28/selected_rf.joblib
```

If the reference is retained, use `outputs/rf_refinement_oct28/reference_rf.joblib` instead. These are research predictions with provisional labels and unconfirmed patient independence.

## Verification and files

`test_refine_expanded_rf.py` verifies equal group mass and sample-weight routing through both target models. An eligible training window was reconstructed from its source PPG; raw-signal inference reproduced the saved-feature model prediction. The inference check is saved with the outputs. Generated model binaries remain local and are excluded from upload.

`refine_expanded_rf.py` reproduces the grouped experiment. Use a new output directory for another run. `summarize_rf_refinement.py` regenerates this report, pressure-range diagnostics and plot. The saved notebook reads results without training. No earlier test result is changed.
