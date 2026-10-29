"""Summarize training-only RF selection and its exploratory validation check."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('outputs/rf_refinement_oct28')


def summarize(output=OUT):
    output=Path(output)
    decision=json.loads((output/'selection.json').read_text());winner=decision['selected']
    metrics=pd.read_csv(output/'training_oof_metrics.csv')
    validation=pd.read_csv(output/'validation_metrics.csv')
    predictions=pd.read_csv(output/'training_oof_predictions.csv.gz')
    bands=[]
    for target in ['sbp','dbp']:
        base=predictions[predictions.model.eq('Expanded reference RF')]
        boundaries=np.quantile(base[target],[0,.25,.5,.75,1])
        for name,frame in predictions.groupby('model'):
            frame=frame.copy();frame['quartile']=pd.cut(frame[target],boundaries,labels=[1,2,3,4],include_lowest=True)
            frame['error']=frame['prediction_'+target]-frame[target]
            for q,part in frame.groupby('quartile',observed=True):
                bands.append(dict(model=name,target=target,quartile=int(q),windows=len(part),
                    label_mean=float(part[target].mean()),prediction_mean=float(part['prediction_'+target].mean()),
                    mae=float(part.error.abs().mean()),bias=float(part.error.mean())))
    bands=pd.DataFrame(bands);bands.to_csv(output/'training_pressure_quartile_errors.csv',index=False)
    held=pd.read_csv(output/'validation_predictions.csv');gains=[]
    if winner!='Expanded reference RF':
        for target in ['sbp','dbp']:
            def errors(frame):
                return frame.assign(ae=(frame['prediction_'+target]-frame[target]).abs()).groupby('overlap_group').ae.mean()
            old=errors(held[held.model.eq('Expanded reference RF')]);new=errors(held[held.model.eq(winner)]).reindex(old.index)
            assert new.notna().all()
            difference=(old-new).to_numpy();rng=np.random.default_rng(20261028)
            bootstrap=difference[rng.integers(len(difference),size=(2000,len(difference)))].mean(axis=1)
            gains.append(dict(target=target,equal_group_mae_gain=float(difference.mean()),ci_low=float(np.quantile(bootstrap,.025)),ci_high=float(np.quantile(bootstrap,.975))))
    pd.DataFrame(gains,columns=['target','equal_group_mae_gain','ci_low','ci_high']).to_csv(output/'validation_paired_group_gains.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,target in zip(axes,['sbp','dbp']):
        names=['Expanded reference RF']+([winner] if winner!='Expanded reference RF' else [])
        for name in names:
            b=bands[bands.model.eq(name)&bands.target.eq(target)].sort_values('quartile')
            ax.plot(b.label_mean,b.prediction_mean,'o-',label=name)
        b=bands[bands.target.eq(target)];low=b.label_mean.min()-3;high=b.label_mean.max()+3
        ax.plot([low,high],[low,high],':',color='gray',label='Prediction equals label')
        ax.set(xlim=(low,high),ylim=(low,high),title=target.upper(),xlabel='Mean label in training pressure quarter (mmHg)',ylabel='Mean out-of-fold prediction (mmHg)')
        ax.legend(fontsize=7)
    fig.suptitle('Does the refinement reduce predictions shrinking toward the middle?')
    fig.tight_layout();fig.savefig(output/'training_shrinkage.png',dpi=160);plt.close(fig)
    text='''# Expanded RF refinement

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
'''
    for name,frame in metrics.groupby('model'):
        v=frame.set_index('target')
        text+=f"| {name} | {v.loc['sbp','overlap_group_macro_mae']:.2f} | {v.loc['dbp','overlap_group_macro_mae']:.2f} |\n"
    text+=f'\nTraining-selected candidate: **{winner}**. These OOF selection scores are optimistic after choosing among candidates; they are not independent final estimates.\n\n## Exploratory validation\n\nSame 857 validation windows; all values are mmHg.\n\n| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE |\n|---|---:|---:|---:|---:|\n'
    for name,frame in validation.groupby('model'):
        v=frame.set_index('target')
        text+=f"| {name} | {v.loc['sbp','mae_mmHg']:.2f} | {v.loc['dbp','mae_mmHg']:.2f} | {v.loc['sbp','rmse_mmHg']:.2f} | {v.loc['dbp','rmse_mmHg']:.2f} |\n"
    text+='''
`validation_paired_group_gains.csv` reports 2,000 paired group resamples for the selected candidate versus the expanded reference. Positive is improvement. The intervals are conditional on these models and do not account for repeatedly inspected validation or all earlier model-development choices.

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
'''
    if winner!='Expanded reference RF':
        old=validation[validation.model.eq('Expanded reference RF')].set_index('target')
        new=validation[validation.model.eq(winner)].set_index('target')
        rmse_note='Both RMSE values also improved in this run.' if (new.rmse_mmHg<old.rmse_mmHg).all() else 'Compare both RMSE values above before replacing the reference.'
        current=bands[bands.model.eq(winner)&bands.target.eq('sbp')].set_index('quartile')
        outcome=f'''## What I would keep

Keep **{winner}** as the next development candidate. Its validation MAE is {new.loc['sbp','mae_mmHg']:.2f}/{new.loc['dbp','mae_mmHg']:.2f}, versus {old.loc['sbp','mae_mmHg']:.2f}/{old.loc['dbp','mae_mmHg']:.2f} for the expanded reference. {rmse_note} The paired equal-group intervals are saved separately; they do not account for the full repeated development process.

The harder problem remains pressure-range generalization. The selected model's lowest/highest training SBP quartiles still have biases about {current.loc[1,'bias']:+.2f}/{current.loc[4,'bias']:+.2f} mmHg. A better average error has not fixed these large misses. I would review representative training signals and labels next instead of expanding into another broad parameter search.

'''
        text=text.replace('## Pressure-range behavior',outcome+'## Pressure-range behavior')
    protocol=json.loads((output/'protocol.json').read_text())
    (output/'candidate_manifest.json').write_text(json.dumps(dict(
        selected=winner,features=protocol['features'],targets=['sbp','dbp'],
        model_path='selected_rf.joblib' if winner!='Expanded reference RF' else 'reference_rf.joblib',
        training_recordings=protocol['training_records'],training_windows=protocol['training_windows'],
        sampling_rate_hz=125,window_samples=625,minimum_context_samples_each_side=125,
        selection=protocol['selection'],validation_status='Exploratory; not a new final test',test_evaluated=False),indent=2))
    Path('RF_REFINEMENT_RESULTS.md').write_text(text)
    return decision

if __name__=='__main__':summarize()
