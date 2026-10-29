"""Focused RF refinement selected entirely on fixed grouped training folds.

Validation is read for scoring only after a training-only selection is written.
No old test data or predictions are accessed. Group-balanced weights are fitted
from training-group counts only and are never predictors.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits
from random_forest_pipeline import forest_model,metric_rows,TARGETS

DATA=Path('outputs/training_learning_curve_oct28')
OUT=Path('outputs/rf_refinement_oct28')
SEED=20261014


def group_weights(groups):
    values=pd.Series(np.asarray(groups))
    counts=values.value_counts()
    weights=1/values.map(counts).to_numpy(dtype=float)
    return weights/weights.mean()


def candidate_models():
    reference=forest_model(seed=SEED).set_params(regressor__forest__max_depth=10)
    deep=forest_model(seed=SEED).set_params(regressor__forest__max_depth=None)
    separate=clone(deep)
    separate.set_params(regressor__forest=MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200,max_depth=None,min_samples_leaf=5,max_features=.8,
        random_state=SEED,n_jobs=2),n_jobs=1))
    return {'Deeper joint RF':(deep,False),'Separate-target RF':(separate,False),
        'Group-balanced reference RF':(reference,True),'Group-balanced separate RF':(clone(separate),True)}


def fit_model(template,X,y,groups,balanced):
    model=clone(template)
    kwargs={'forest__sample_weight':group_weights(groups)} if balanced else {}
    with threadpool_limits(limits=2):model.fit(X,y,**kwargs)
    return model


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    meta=pd.read_csv(DATA/'rows_and_labels.csv')
    assert set(meta.split)=={'train','validation'}
    features=pd.read_csv(DATA/'original_plus_pulse_features.csv')
    manifest=json.loads((DATA/'candidate_manifest.json').read_text())
    assert list(features.columns)==manifest['features']
    assert len(meta)==len(features)
    X=features[meta.split.eq('train')].replace([np.inf,-np.inf],np.nan)
    train=meta[meta.split.eq('train')].copy();y=train[TARGETS].to_numpy()
    folds=pd.read_csv(DATA/'group_folds.csv').set_index('overlap_group').fold
    assigned=train.overlap_group.map(folds).to_numpy()
    assert set(assigned)=={0,1,2}
    models=candidate_models()
    protocol=dict(date='2026-10-28',training_records=int(train.record_key.nunique()),training_windows=len(train),
        features=list(features.columns),folds='Same three full-expansion overlap-group folds as learning curve',
        seed=SEED,test_evaluated=False,
        candidates={name:dict(estimator=repr(estimator),group_balanced=balanced) for name,(estimator,balanced) in models.items()},
        weighting='Inverse window count per overlap group, normalized to mean one; recomputed within each training fold',
        selection='Lowest mean of target-wise group-macro OOF MAE; replacement needs >=0.05 mmHg mean gain and no target-wise group-macro MAE regression',
        validation='Score reference and training-selected replacement only, after selection is saved; exploratory',
        caution='Selection OOF scores are optimistic after choosing candidates; not independent final scores')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    old_path=DATA/'predictions.csv.gz'
    saved=pd.read_csv(old_path)
    reference=saved.query('level=="Full expansion" and variant=="Original + pulse" and model=="RF" and split=="training_oof"').set_index('window_key').loc[train.window_key]
    np.testing.assert_allclose(reference[TARGETS],y)
    baseline=reference[['prediction_sbp','prediction_dbp']].to_numpy()
    metrics=[];frames=[]
    for name,prediction in [('Expanded reference RF',baseline)]:
        metrics.extend(metric_rows(train,y,prediction,name))
        frame=train[['window_key','record_key','overlap_group','part']+TARGETS].copy();frame['model']=name
        for i,t in enumerate(TARGETS):frame['prediction_'+t]=prediction[:,i]
        frames.append(frame)
    for name,(template,balanced) in models.items():
        print('Training-only folds:',name,flush=True)
        prediction=np.full(y.shape,np.nan)
        for fold in range(3):
            inside=np.flatnonzero(assigned!=fold);held=np.flatnonzero(assigned==fold)
            assert set(train.overlap_group.iloc[inside]).isdisjoint(train.overlap_group.iloc[held])
            model=fit_model(template,X.iloc[inside],y[inside],train.overlap_group.iloc[inside],balanced)
            with threadpool_limits(limits=2):prediction[held]=model.predict(X.iloc[held])
        assert np.isfinite(prediction).all()
        metrics.extend(metric_rows(train,y,prediction,name))
        frame=train[['window_key','record_key','overlap_group','part']+TARGETS].copy();frame['model']=name
        for i,t in enumerate(TARGETS):frame['prediction_'+t]=prediction[:,i]
        frames.append(frame)
        pd.DataFrame(metrics).to_csv(OUT/'training_oof_metrics.csv',index=False)
        print(pd.DataFrame(metrics).query('model==@name')[['target','mae_mmHg','overlap_group_macro_mae']].round(3).to_string(index=False),flush=True)
    metrics=pd.DataFrame(metrics);all_predictions=pd.concat(frames,ignore_index=True)
    all_predictions.to_csv(OUT/'training_oof_predictions.csv.gz',index=False,compression='gzip')
    scores=metrics.pivot(index='model',columns='target',values='overlap_group_macro_mae')
    reference_scores=scores.loc['Expanded reference RF'];ranking=scores.mean(axis=1).sort_values()
    winner='Expanded reference RF'
    for name in ranking.index:
        if name=='Expanded reference RF':continue
        if (reference_scores-scores.loc[name]).mean()>=.05 and (scores.loc[name]<=reference_scores).all():
            winner=name;break
    decision=dict(selected=winner,reference='Expanded reference RF',basis=protocol['selection'],
        group_macro_oof_scores=scores.to_dict(orient='index'),test_evaluated=False,
        status='Training-only decision written before validation prediction')
    (OUT/'selection.json').write_text(json.dumps(decision,indent=2))
    print('Training-selected:',winner,flush=True)
    reference_path=DATA/'original_plus_pulse_rf.joblib'
    if reference_path.exists():
        reference_model=joblib.load(reference_path)
    else:
        reference_model=fit_model(forest_model(seed=SEED).set_params(regressor__forest__max_depth=10),X,y,train.overlap_group,False)
    selected={'Expanded reference RF':reference_model}
    joblib.dump(reference_model,OUT/'reference_rf.joblib')
    if winner!='Expanded reference RF':
        template,balanced=models[winner]
        selected[winner]=fit_model(template,X,y,train.overlap_group,balanced)
        joblib.dump(selected[winner],OUT/'selected_rf.joblib')
    # Open held-out values only after the training-only decision above.
    validation=meta[meta.split.eq('validation')].copy()
    V=features.loc[validation.index].replace([np.inf,-np.inf],np.nan);truth=validation[TARGETS].to_numpy()
    rows=[];outputs=[]
    for name,model in selected.items():
        with threadpool_limits(limits=2):prediction=model.predict(V)
        assert np.isfinite(prediction).all()
        rows.extend(metric_rows(validation,truth,prediction,name))
        frame=validation[['window_key','record_key','overlap_group','part']+TARGETS].copy();frame['model']=name
        for i,t in enumerate(TARGETS):frame['prediction_'+t]=prediction[:,i]
        outputs.append(frame)
    pd.DataFrame(rows).to_csv(OUT/'validation_metrics.csv',index=False)
    pd.concat(outputs,ignore_index=True).to_csv(OUT/'validation_predictions.csv',index=False)
    from summarize_rf_refinement import summarize
    summarize(OUT)
    (OUT/'COMPLETED.json').write_text(json.dumps(dict(status='complete',selected=winner,test_evaluated=False)))
    print(pd.DataFrame(rows)[['model','target','mae_mmHg','rmse_mmHg']].round(3).to_string(index=False),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();DATA=args.data_dir;OUT=args.output_dir
    main()
