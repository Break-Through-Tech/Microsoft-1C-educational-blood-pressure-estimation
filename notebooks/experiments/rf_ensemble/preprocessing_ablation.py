"""Controlled PPG preprocessing comparison; original labels/rows/groups fixed.

Only training and exploratory validation are processed. Test rows are excluded
before any waveform access. Models and hyperparameters are fixed across variants.
"""
from pathlib import Path
import json, hashlib, warnings
import h5py, joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from random_forest_pipeline import load_data, forest_model, metric_rows, TARGETS
from expanded_data_analysis import extract_features

OUT=Path('outputs/preprocessing_ablation_oct28')
DATA=Path('outputs/expanded_eda_oct14')
FS=125


def zscore(x):
    std=np.std(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.full(len(x),np.nan)
    return (x-np.mean(x))/std


def pulse_features(signal, peaks):
    """Use only complete foot-peak-foot pulses, measured in a z-scored window."""
    signal=zscore(signal)
    rows=[]
    for previous, peak, following in zip(peaks[:-2],peaks[1:-1],peaks[2:]):
        foot=previous+int(np.argmin(signal[previous:peak+1]))
        end=peak+int(np.argmin(signal[peak:following+1]))
        if not foot < peak < end:
            continue
        pulse=signal[foot:end+1]
        baseline=np.linspace(signal[foot],signal[end],len(pulse))
        corrected=pulse-baseline
        height=corrected[peak-foot]
        if not np.isfinite(height) or height <= 0:
            continue
        rise=(peak-foot)/FS; fall=(end-peak)/FS
        values=dict(rise_s=rise,fall_s=fall,rise_fall_ratio=rise/fall,
            duration_s=(end-foot)/FS,height=height,
            upslope=np.max(np.diff(signal[foot:peak+1]))*FS,
            area=np.trapezoid(corrected,dx=1/FS),rise_fraction=rise/(rise+fall))
        for fraction in [.25,.5,.75]:
            mask=corrected>=fraction*height
            # Only the contiguous threshold segment containing the detected peak.
            left=right=peak-foot
            while left>0 and mask[left-1]: left-=1
            while right<len(mask)-1 and mask[right+1]: right+=1
            values['width_'+str(int(fraction*100))+'_s']=(right-left)/FS
        rows.append(values)
    names=['rise_s','fall_s','rise_fall_ratio','duration_s','height','upslope','area','rise_fraction','width_25_s','width_50_s','width_75_s']
    result={}
    for name in names:
        a=np.array([r[name] for r in rows])
        result['pulse_'+name+'_median']=float(np.median(a)) if len(a)>=2 else np.nan
        result['pulse_'+name+'_iqr']=float(np.ptp(np.quantile(a,[.25,.75]))) if len(a)>=2 else np.nan
    result['pulse_rate_bpm']=float(60/np.median(np.diff(peaks)/FS)) if len(peaks)>1 else np.nan
    return result,len(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    table,features=load_data(DATA)
    selected=table[table.eligible & table.split.isin(['train','validation'])].copy().reset_index(drop=True)
    assert not selected.split.eq('test').any()
    protocol=dict(date='2026-10-28',rows=len(selected),test_processed=False,
        data_sha256=hashlib.sha256((DATA/'expanded_features_and_split.csv').read_bytes()).hexdigest(),
        selection='Compare fixed-model variants by training-only grouped out-of-fold errors; validation exploratory',
        controls='Same eligible rows, labels, groups, folds and fixed models across variants; no quality gate or target changes',
        variants=['Original 37','Normalized 37','NK pulse shape','Original + NK pulse shape','Bandpass pulse shape'],
        filtering='NK ppg_clean default vs fourth-order Butterworth 0.5–8Hz, both full-record. Same NK peak positions for both pulse variants to isolate filtering effects.',
        normalization='Per-window z-score; no cross-window fitted statistics',
        pulse_definition='Minimum between successive fixed peaks defines feet; complete foot-peak-foot only; >=2 pulses otherwise shape values missing',
        missing_policy='Keep all reference rows; median imputation fitted on each training fold. Record missing pulse counts.',
        model_policy='RF original configuration; boosting prior fixed configuration. No parameter search in this experiment.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    norm_rows={};nk_rows={};bp_rows={};audit=[]
    sos=butter(4,[.5,8],fs=FS,btype='bandpass',output='sos')
    jobs=list(selected.groupby(['part','record_id']))
    for number,((part,rid),rows) in enumerate(jobs,1):
        with h5py.File(Path('data')/(part+'.mat'),'r') as h:
            raw=np.array(h[h[part][int(rid),0]][:,0],dtype=float)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            clean=nk.ppg_clean(raw,sampling_rate=FS)
            peaks=np.asarray(nk.ppg_findpeaks(clean,sampling_rate=FS)['PPG_Peaks'],dtype=int)
            band=sosfiltfilt(sos,raw)
        for idx,row in rows.iterrows():
            start=int(row.start_sample);stop=start+625
            segment=raw[start:stop]
            digest=hashlib.sha256(np.asarray(segment,dtype='<f8').tobytes()).hexdigest()
            assert digest==row.raw_ppg_sha256, row.window_key
            pp=peaks[(peaks>=start)&(peaks<stop)]-start
            # Existing statistics and peak-to-peak summaries, with scale removed.
            norm_rows[idx]=extract_features(zscore(segment),zscore(clean[start:stop]),pp)
            nk_rows[idx],n=pulse_features(clean[start:stop],pp)
            bp_rows[idx],b=pulse_features(band[start:stop],pp)
            audit.append(dict(window_key=row.window_key,split=row.split,nk_complete_pulses=n,
                bandpass_complete_pulses=b,record_warnings=' | '.join(str(w.message) for w in captured)))
        if number%100==0 or number==len(jobs): print('Processed recordings',number,'/',len(jobs),flush=True)
    original=selected[features].copy()
    normalized=pd.DataFrame.from_dict(norm_rows,orient='index').sort_index()[features]
    # These statistics are constants by construction; avoid binning floating-point noise.
    for prefix in ['ppg_raw_', 'ppg_clean_']:
        normalized[prefix+'mean']=0.0
        normalized[prefix+'std']=1.0
    pulse=pd.DataFrame.from_dict(nk_rows,orient='index').sort_index()
    band=pd.DataFrame.from_dict(bp_rows,orient='index').sort_index()
    variants={'Original 37':original,'Normalized 37':normalized,'NK pulse shape':pulse,
        'Original + NK pulse shape':pd.concat([original,pulse],axis=1),'Bandpass pulse shape':band}
    for name,frame in variants.items():
        frame.to_csv(OUT/(name.lower().replace(' ','_').replace('+','plus')+'_features.csv'),index=False)
    selected[['window_key','record_key','overlap_group','part','split']+TARGETS].to_csv(OUT/'rows_and_labels.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'pulse_audit.csv',index=False)
    rf=forest_model(seed=20261014).set_params(regressor__forest__max_depth=10)
    boost=TransformedTargetRegressor(regressor=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('model',MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,
            max_leaf_nodes=7,l2_regularization=10,min_samples_leaf=20,early_stopping=False,random_state=20261028),n_jobs=1))]),transformer=StandardScaler())
    train=selected[selected.split.eq('train')];val=selected[selected.split.eq('validation')]
    y=train[TARGETS].to_numpy();vy=val[TARGETS].to_numpy()
    folds=list(GroupKFold(3).split(train,y,train.overlap_group))
    assignments=train[['window_key','overlap_group']].copy();assignments['fold']=-1
    for i,(a,b) in enumerate(folds):
        assert set(train.overlap_group.iloc[a]).isdisjoint(train.overlap_group.iloc[b])
        assignments.iloc[b,assignments.columns.get_loc('fold')]=i
    assignments.to_csv(OUT/'fold_assignments.csv',index=False)
    all_metrics=[];predictions=[]
    for name,frame in variants.items():
        frame=frame.replace([np.inf,-np.inf],np.nan)
        X=frame.loc[train.index];V=frame.loc[val.index]
        for model_name,template in [('RF',rf),('Boosting',boost)]:
            print('Fitting',name,model_name,flush=True)
            oof=np.full(y.shape,np.nan)
            for a,b in folds:
                assert X.iloc[a].notna().any().all(), 'Entirely missing feature in fold'
                model=clone(template)
                with threadpool_limits(limits=2):
                    model.fit(X.iloc[a],y[a]);oof[b]=model.predict(X.iloc[b])
            model=clone(template)
            with threadpool_limits(limits=2):
                model.fit(X,y);vp=model.predict(V)
            joblib.dump(model,OUT/(name.lower().replace(' ','_').replace('+','plus')+'_'+model_name.lower()+'.joblib'))
            for split,meta,truth,guesses in [('training_oof',train,y,oof),('validation',val,vy,vp)]:
                assert np.isfinite(guesses).all()
                rows=metric_rows(meta,truth,guesses,model_name)
                all_metrics.extend(dict(r,variant=name,split=split,features=len(frame.columns)) for r in rows)
                result=meta[['window_key','record_key','overlap_group','part']].copy()
                result['split']=split;result['variant']=name;result['model']=model_name
                for i,t in enumerate(TARGETS):result[t]=truth[:,i];result['prediction_'+t]=guesses[:,i]
                predictions.append(result)
            pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
    # Training-only out-of-fold constants, fit independently in each fold.
    for strategy in ['mean','median']:
        oof=np.full(y.shape,np.nan)
        for a,b in folds:oof[b]=getattr(np,strategy)(y[a],axis=0)
        vp=np.broadcast_to(getattr(np,strategy)(y,axis=0),vy.shape)
        for split,meta,truth,guesses in [('training_oof',train,y,oof),('validation',val,vy,vp)]:
            all_metrics.extend(dict(r,variant='Constant',split=split,features=0) for r in metric_rows(meta,truth,guesses,strategy))
    pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    from summarize_preprocessing_comparison import summarize
    summarize(OUT)
    (OUT/'COMPLETED.json').write_text(json.dumps(dict(status='complete',test_processed=False)))
    print(pd.DataFrame(all_metrics)[['variant','model','split','target','mae_mmHg','rmse_mmHg','overlap_group_macro_mae']].round(3).to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    OUT=parser.parse_args().output_dir
    main()
