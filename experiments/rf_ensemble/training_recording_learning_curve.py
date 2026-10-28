"""Nested training-recording learning curve with fixed exploratory validation.

Uses the expanded, overlap-screened training table. No old test features or
predictions are read. Model settings and features are fixed before scoring.
"""
from pathlib import Path
import hashlib,json,warnings
import h5py,joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from random_forest_pipeline import load_data,forest_model,metric_rows,TARGETS
from expanded_data_analysis import extract_features
from preprocessing_ablation import pulse_features,zscore

DATA=Path('outputs/training_expansion_oct28')
CACHE=Path('outputs/preprocessing_ablation_oct28')
OUT=Path('outputs/training_learning_curve_oct28')
SEED=20261028


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    table,features=load_data(DATA)
    assert set(table.split)=={'train','validation'}
    table=table.reset_index(drop=True)
    protocol=dict(seed=SEED,test_evaluated=False,models='Original RF and fixed histogram boosting',
        variants=['Original','Normalized','Original + pulse'],
        levels='Nested original groups, plus half of new groups, plus all new groups; no target-based sampling',
        validation='Fixed original windows; exploratory and repeatedly inspected',
        folds='GroupKFold on complete training pool; restrict each fixed fold to nested subsets',
        feature_policy='No new feature choice, model tuning, pressure bounds or quality gate changes',
        data_sha256=hashlib.sha256((DATA/'expanded_features_and_split.csv').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cached_meta=pd.read_csv(CACHE/'rows_and_labels.csv')
    norm_cache=pd.read_csv(CACHE/'normalized_37_features.csv');norm_cache.index=cached_meta.window_key
    pulse_cache=pd.read_csv(CACHE/'nk_pulse_shape_features.csv');pulse_cache.index=cached_meta.window_key
    normalized={};pulse={};audit=[]
    old=table.window_key.isin(cached_meta.window_key)
    for idx,row in table[old].iterrows():
        normalized[idx]=norm_cache.loc[row.window_key].to_dict()
        pulse[idx]=pulse_cache.loc[row.window_key].to_dict()
    jobs=list(table[~old].groupby(['part','record_id']))
    for number,((part,rid),rows) in enumerate(jobs,1):
        with h5py.File(Path('data')/(part+'.mat'),'r') as handle:
            raw=np.array(handle[handle[part][int(rid),0]][:,0],dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clean=nk.ppg_clean(raw,sampling_rate=125)
            peaks=np.asarray(nk.ppg_findpeaks(clean,sampling_rate=125)['PPG_Peaks'],dtype=int)
        for idx,row in rows.iterrows():
            start=int(row.start_sample);stop=start+625;x=raw[start:stop];c=clean[start:stop]
            assert hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()==row.raw_ppg_sha256
            pp=peaks[(peaks>=start)&(peaks<stop)]-start
            normalized[idx]=extract_features(zscore(x),zscore(c),pp)
            for prefix in ['ppg_raw_','ppg_clean_']:
                normalized[idx][prefix+'mean']=0.;normalized[idx][prefix+'std']=1.
            pulse[idx],count=pulse_features(c,pp)
            audit.append(dict(window_key=row.window_key,complete_pulses=count,
                warnings=' | '.join(str(w.message) for w in caught)))
        if number%100==0 or number==len(jobs):print('New feature records',number,'/',len(jobs),flush=True)
    norm=pd.DataFrame.from_dict(normalized,orient='index').sort_index()[features]
    shapes=pd.DataFrame.from_dict(pulse,orient='index').sort_index()
    variants={'Original':table[features],'Normalized':norm,'Original + pulse':pd.concat([table[features],shapes],axis=1)}
    for name,frame in variants.items():frame.to_csv(OUT/(name.lower().replace(' ','_').replace('+','plus')+'_features.csv'),index=False)
    pd.DataFrame(audit).to_csv(OUT/'new_pulse_audit.csv',index=False)
    table[['window_key','record_key','overlap_group','part','split','training_origin']+TARGETS].to_csv(OUT/'rows_and_labels.csv',index=False)
    training=table[table.split.eq('train')];validation=table[table.split.eq('validation')]
    old_groups=set(training.loc[training.training_origin.eq('original'),'overlap_group'])
    new_groups=np.array(sorted(set(training.overlap_group)-old_groups))
    np.random.default_rng(SEED).shuffle(new_groups)
    levels={'Original training':old_groups,'Half expansion':old_groups|set(new_groups[:len(new_groups)//2]),'Full expansion':set(training.overlap_group)}
    pd.DataFrame([dict(overlap_group=g,level=name) for name,groups in levels.items() for g in sorted(groups)]).to_csv(OUT/'nested_groups.csv',index=False)
    fold_map={}
    for fold,(_,held) in enumerate(GroupKFold(3).split(training,training[TARGETS],training.overlap_group)):
        for g in training.overlap_group.iloc[held]:fold_map[g]=fold
    pd.DataFrame([dict(overlap_group=g,fold=f) for g,f in fold_map.items()]).to_csv(OUT/'group_folds.csv',index=False)
    rf=forest_model(seed=20261014).set_params(regressor__forest__max_depth=10)
    boost=TransformedTargetRegressor(regressor=Pipeline([('imputer',SimpleImputer(strategy='median')),
        ('model',MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=7,
        l2_regularization=10,min_samples_leaf=20,early_stopping=False,random_state=SEED),n_jobs=1))]),transformer=StandardScaler())
    metrics=[];predictions=[];counts=[]
    for level,groups in levels.items():
        train=training[training.overlap_group.isin(groups)];y=train[TARGETS].to_numpy();vy=validation[TARGETS].to_numpy()
        folds=np.array([fold_map[g] for g in train.overlap_group])
        counts.append(dict(level=level,windows=len(train),recordings=train.record_key.nunique(),groups=train.overlap_group.nunique()))
        pd.DataFrame(counts).to_csv(OUT/'training_counts.csv',index=False)
        for variant,features_table in variants.items():
            X=features_table.loc[train.index].replace([np.inf,-np.inf],np.nan)
            V=features_table.loc[validation.index].replace([np.inf,-np.inf],np.nan)
            for model_name,template in [('RF',rf),('Boosting',boost)]:
                print(level,variant,model_name,len(train),'windows',flush=True)
                oof=np.full(y.shape,np.nan)
                for fold in range(3):
                    inside=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
                    assert set(train.overlap_group.iloc[inside]).isdisjoint(train.overlap_group.iloc[held])
                    assert X.iloc[inside].notna().any().all()
                    model=clone(template)
                    with threadpool_limits(limits=2):model.fit(X.iloc[inside],y[inside]);oof[held]=model.predict(X.iloc[held])
                model=clone(template)
                with threadpool_limits(limits=2):model.fit(X,y);vp=model.predict(V)
                if level=='Full expansion':joblib.dump(model,OUT/(variant.lower().replace(' ','_').replace('+','plus')+'_'+model_name.lower()+'.joblib'))
                for split,meta,truth,guesses in [('training_oof',train,y,oof),('validation',validation,vy,vp)]:
                    assert np.isfinite(guesses).all()
                    metrics.extend(dict(r,level=level,variant=variant,split=split,training_records=int(train.record_key.nunique()),training_windows=len(train)) for r in metric_rows(meta,truth,guesses,model_name))
                    frame=meta[['window_key','record_key','overlap_group','part']].copy()
                    frame['level']=level;frame['variant']=variant;frame['model']=model_name;frame['split']=split
                    for i,t in enumerate(TARGETS):frame[t]=truth[:,i];frame['prediction_'+t]=guesses[:,i]
                    predictions.append(frame)
                pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
        # Recompute constants for each training subset; no validation fit.
        for strategy in ['mean','median']:
            vp=np.broadcast_to(getattr(np,strategy)(y,axis=0),vy.shape)
            oof=np.full(y.shape,np.nan)
            for fold in range(3):oof[folds==fold]=getattr(np,strategy)(y[folds!=fold],axis=0)
            for split,meta,truth,guesses in [('training_oof',train,y,oof),('validation',validation,vy,vp)]:
                metrics.extend(dict(r,level=level,variant='Constant',split=split,training_records=int(train.record_key.nunique()),training_windows=len(train)) for r in metric_rows(meta,truth,guesses,strategy))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    from summarize_training_learning_curve import summarize
    summarize(OUT)
    (OUT/'COMPLETED.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    print(pd.DataFrame(metrics).query('split=="validation"')[['level','variant','model','target','mae_mmHg','rmse_mmHg']].round(3).to_string(index=False),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();DATA=args.data_dir;OUT=args.output_dir
    main()
