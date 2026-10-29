"""Summarize the fixed-validation training-recording learning curve."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('outputs/training_learning_curve_oct28')
DATA=Path('outputs/training_expansion_oct28')


def summarize(output=OUT):
    output=Path(output)
    m=pd.read_csv(output/'metrics.csv')
    predictions_path=output/'predictions.csv'
    if not predictions_path.exists():predictions_path=output/'predictions.csv.gz'
    p=pd.read_csv(predictions_path)
    counts=pd.read_csv(output/'training_counts.csv')
    from random_forest_pipeline import metric_rows
    # Evaluate OOF predictions on the SAME original training windows at every size.
    # Their group fold never changes, so extra records cannot enter a held-out group.
    original_keys=set(p.loc[p.split.eq('training_oof') & p.level.eq('Original training'),'window_key'])
    fixed_rows=[]
    for (level,variant,model),frame in p[p.split.eq('training_oof')].groupby(['level','variant','model']):
        frame=frame[frame.window_key.isin(original_keys)]
        assert set(frame.window_key)==original_keys and not frame.window_key.duplicated().any()
        truth=frame[['sbp','dbp']].to_numpy();guesses=frame[['prediction_sbp','prediction_dbp']].to_numpy()
        fixed_rows.extend(dict(r,level=level,variant=variant) for r in metric_rows(frame,truth,guesses,model))
    pd.DataFrame(fixed_rows).to_csv(output/'fixed_original_oof_metrics.csv',index=False)
    rows=[]
    val=p[p.split.eq('validation')]
    for model in ['RF','Boosting']:
        for variant in ['Original','Normalized','Original + pulse']:
            sub=val[val.model.eq(model)&val.variant.eq(variant)]
            old=sub[sub.level.eq('Original training')].set_index('window_key').sort_index()
            for level in ['Half expansion','Full expansion']:
                new=sub[sub.level.eq(level)].set_index('window_key').sort_index()
                assert old.index.equals(new.index)
                for target in ['sbp','dbp']:
                    assert np.array_equal(old[target].to_numpy(),new[target].to_numpy())
                    old_ae=(old['prediction_'+target]-old[target]).abs().groupby(old.overlap_group).mean()
                    new_ae=(new['prediction_'+target]-new[target]).abs().groupby(new.overlap_group).mean().reindex(old_ae.index)
                    assert new_ae.notna().all()
                    gains=(old_ae-new_ae).to_numpy();rng=np.random.default_rng(20261028)
                    boot=gains[rng.integers(len(gains),size=(2000,len(gains)))].mean(axis=1)
                    rows.append(dict(model=model,variant=variant,level=level,target=target,
                        equal_group_mae_gain=float(gains.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(rows).to_csv(output/'paired_expansion_gains.csv',index=False)
    levels=['Original training','Half expansion','Full expansion']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,model in enumerate(['RF','Boosting']):
        for j,target in enumerate(['sbp','dbp']):
            ax=axes[i,j]
            for variant,color in [('Original','#63778c'),('Normalized','#3c989b'),('Original + pulse','#a88252')]:
                v=m.query('split=="validation" and model==@model and target==@target and variant==@variant').set_index('level').reindex(levels)
                ax.plot(v.training_records,v.mae_mmHg,'o-',label=variant,color=color)
                # Exact values are in the report table; omit overlapping labels.
            baseline=m.query('split=="validation" and model=="mean" and target==@target').set_index('level').reindex(levels)
            ax.plot(baseline.training_records,baseline.mae_mmHg,':',color='gray',label='Mean baseline')
            ax.set_title(model+' — '+target.upper());ax.set_xlabel('Training recordings');ax.set_ylabel('Validation MAE (mmHg)');ax.legend(fontsize=8)
    fig.suptitle('More training recordings, same fixed validation windows')
    fig.tight_layout();fig.savefig(output/'learning_curve.png',dpi=160);plt.close(fig)
    lines=['| Training set | Recordings | Windows |','|---|---:|---:|']
    for r in counts.itertuples():lines.append(f'| {r.level} | {r.recordings} | {r.windows} |')
    text='''# Training-recording learning curve

October 28, 2026. This experiment adds training recordings while keeping the original validation windows fixed. All results are exploratory development results; the old test was not scored.

## Training size

'''+ '\n'.join(lines)+'''\n
Each new source recording was sampled without looking at pressure values. The same quality gates and pressure-label method were used as before. The expansion samples 250 source recordings per Part, up to 12 windows per recording. The half/full subsets are nested, with seeded new-group order. Duplicate and shifted-overlap detection was rerun before training. Training records connected to held-out identities were excluded. The preparation summary records exclusions and retention.

## Validation MAE

All entries are mmHg. The same fixed RF and boosting settings are used at every size and with every feature set. Lower is better.

| Training set | Features | Model | SBP MAE | DBP MAE |
|---|---|---|---:|---:|
'''
    for level in levels:
        for variant in ['Original','Normalized','Original + pulse','Constant']:
            for model in (['mean','median'] if variant=='Constant' else ['RF','Boosting']):
                v=m.query('split=="validation" and level==@level and variant==@variant and model==@model').set_index('target')
                text+=f"| {level} | {variant} | {model} | {v.loc['sbp','mae_mmHg']:.2f} | {v.loc['dbp','mae_mmHg']:.2f} |\n"
    text+='''
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
'''
    # Choose a candidate from fixed-population training OOF, not validation rank.
    fixed=pd.DataFrame(fixed_rows)
    ranking=fixed[fixed.level.eq('Full expansion')].groupby(['variant','model']).overlap_group_macro_mae.mean().sort_values()
    (output/'training_candidate_ranking.json').write_text(json.dumps([dict(variant=k[0],model=k[1],mean_target_equal_group_oof_mae=float(v)) for k,v in ranking.items()],indent=2))
    best_variant,best_model=ranking.index[0]
    slug=best_variant.lower().replace(' ','_').replace('+','plus')
    cols=pd.read_csv(output/(slug+'_features.csv'),nrows=0).columns.tolist()
    full_count=counts.set_index('level').loc['Full expansion']
    (output/'candidate_manifest.json').write_text(json.dumps(dict(
        candidate=best_variant+' / '+best_model,model_path=slug+'_'+best_model.lower()+'.joblib',
        features=cols,training_recordings=int(full_count.recordings),training_windows=int(full_count.windows),
        selection='Lowest mean target/group OOF MAE on fixed original training windows among six full-expansion candidates',
        validation_status='Exploratory; repeatedly inspected; no new final test',old_test_evaluated_again=False),indent=2))
    findings='''## What we learned

The expanded RF with original-plus-pulse features is the most promising candidate. It has the lowest average target/group OOF error on the fixed original training windows. Its validation MAE is 16.12 SBP / 8.55 DBP mmHg, compared with 16.86/8.83 for the original-size RF with original features. Validation RMSE improves from 22.44/12.16 to 21.49/11.65. The full-size mean baseline remains about 18.70/10.02 MAE, so a shifted constant prediction does not explain the gains.

Normalized RF has a nearly identical systolic validation MAE of 16.10, but a higher diastolic MAE of 8.68. I would carry forward the added-pulse RF because its fixed-population training OOF comparison is stronger, rather than pick a model just for its lowest validation systolic number.

More training recordings gave a steadier improvement than the previous parameter changes. The gain is still modest, and most paired validation intervals for expansion include zero. This is a better development candidate, not evidence of final accuracy. Keep the original model and all earlier test results as references. The next useful investigation is a representative training-only label/signal review before deciding whether more data, label refinements or further features are worthwhile.

'''
    text=text.replace('## How to read this experiment',findings+'## How to read this experiment')
    Path('TRAINING_EXPANSION_RESULTS.md').write_text(text)
    return m

if __name__=='__main__':summarize()
