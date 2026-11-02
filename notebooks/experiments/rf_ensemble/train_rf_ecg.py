"""Refit the RF-only PPG/ECG comparison from frozen feature tables.

Use a new output directory. All imputation and target scaling are fitted inside
training folds. The old test is absent from these inputs.
"""
from pathlib import Path
import argparse,json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from standard_bp_metrics import evaluate_predictions


def make_model():
    return TransformedTargetRegressor(regressor=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('forest',RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_leaf=5,
            max_features=.8,random_state=20261014,n_jobs=2))]),transformer=StandardScaler())


def train(data,output):
    output.mkdir(parents=True,exist_ok=False)
    cohort=data/'matched_cohort.csv' if (data/'matched_cohort.csv').exists() else data/'rows_and_labels.csv'
    meta=pd.read_csv(cohort)
    assert set(meta.split)=={'train','validation'}
    for identity in ['record_key','overlap_group']:
        assert meta.groupby(identity).split.nunique().max()==1
    ppg=pd.read_csv(data/'ppg_features.csv');ecg=pd.read_csv(data/'ecg_features.csv')
    assert len(ppg)==len(ecg) and meta.feature_row.max()<len(ppg)
    folds=pd.read_csv(data/'training_folds.csv').set_index('window_key').fold
    t=meta[meta.split.eq('train')];v=meta[meta.split.eq('validation')]
    y=t[['sbp','dbp']].to_numpy();assigned=t.window_key.map(folds).to_numpy()
    assert set(assigned)=={0,1,2}
    means=t[['sbp','dbp']].mean().to_dict();metrics=[];predictions=[]
    (output/'settings.json').write_text(json.dumps(dict(model=repr(make_model()),test_evaluated=False,
        training_windows=len(t),validation_windows=len(v),baseline_means=means),indent=2))
    for modality,frame in [('PPG',ppg),('PPG + ECG',pd.concat([ppg,ecg],axis=1))]:
        assert all(c.startswith(('ppg_','vpg_','apg_','pulse_','ecg_')) for c in frame)
        X=frame.iloc[t.feature_row].replace([np.inf,-np.inf],np.nan)
        V=frame.iloc[v.feature_row].replace([np.inf,-np.inf],np.nan)
        oof=np.full(y.shape,np.nan)
        for fold in range(3):
            inside=np.flatnonzero(assigned!=fold);held=np.flatnonzero(assigned==fold)
            assert set(t.overlap_group.iloc[inside]).isdisjoint(t.overlap_group.iloc[held])
            model=make_model()
            with threadpool_limits(limits=2):model.fit(X.iloc[inside],y[inside]);oof[held]=model.predict(X.iloc[held])
        model=make_model()
        with threadpool_limits(limits=2):model.fit(X,y);guess=model.predict(V)
        slug=modality.lower().replace(' + ','_and_')
        joblib.dump(model,output/(slug+'_random_forest.joblib'))
        for split,metadata,pred in [('training_oof',t,oof),('validation',v,guess)]:
            assert np.isfinite(pred).all()
            result=metadata[['window_key','record_key','overlap_group','sbp','dbp']].copy()
            result['model']='Random Forest'
            for i,target in enumerate(['sbp','dbp']):result['prediction_'+target]=pred[:,i]
            scores=evaluate_predictions(result,means);scores['split']=split;scores['modality']=modality
            if split=='training_oof':scores[['mean_baseline_mae','mae_fraction_of_mean_baseline']]=np.nan
            metrics.append(scores);result['split']=split;result['modality']=modality;predictions.append(result)
    pd.concat(metrics,ignore_index=True).to_csv(output/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'predictions.csv.gz',index=False,compression='gzip')
    print(pd.concat(metrics).query('split=="validation"')[['modality','target','mae_mmHg','rmse_mmHg','within_10_mmHg_pct']].round(3).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('outputs/rf_ecg_nov1'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/rf_ecg_new_run'))
    args=parser.parse_args();train(args.data_dir,args.output_dir)
