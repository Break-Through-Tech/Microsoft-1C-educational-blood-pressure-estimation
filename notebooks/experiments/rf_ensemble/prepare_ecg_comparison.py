"""Add ECG features to the locked PPG training/validation cohort.

No ABP values enter predictors. No old test waveforms are read. ECG failures
stay in the cohort with missing ECG features so the comparison uses identical
rows rather than dropping harder examples from the ECG experiment.
"""
from pathlib import Path
import hashlib,json,warnings
import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew,kurtosis,iqr

BASE=Path('outputs/rf_ecg_nov1')
OUT=Path('outputs/rf_ecg_features_new')
FS=125


def describe(x,prefix):
    return dict(zip([prefix+n for n in ['mean','std','iqr','skew','kurtosis','p05','p95']],
        [float(np.mean(x)),float(np.std(x)),float(iqr(x)),float(skew(x)),float(kurtosis(x)),float(np.quantile(x,.05)),float(np.quantile(x,.95))]))


def ecg_features(raw,clean,rpeaks,ppg_peaks,start):
    stop=start+625;x=raw[start:stop];c=clean[start:stop]
    f=describe(x,'ecg_raw_');f.update(describe(c,'ecg_clean_'))
    z=(c-c.mean())/c.std() if c.std()>1e-12 else np.full(len(c),np.nan)
    for name,value in [('min',np.min(z)),('max',np.max(z)),('median',np.median(z)),('iqr',iqr(z)),('skew',skew(z)),('kurtosis',kurtosis(z))]:f['ecg_z_'+name]=float(value)
    rr=rpeaks[(rpeaks>=start)&(rpeaks<stop)]
    gaps=np.diff(rr)/FS
    f['ecg_rate_bpm']=float(60/np.median(gaps)) if len(gaps) else np.nan
    f['ecg_rr_std_s']=float(np.std(gaps)) if len(gaps) else np.nan
    f['ecg_rr_cv']=float(np.std(gaps)/np.mean(gaps)) if len(gaps) else np.nan
    heights=z[rr-start]
    f['ecg_r_height_mean']=float(np.mean(heights)) if len(heights) else np.nan
    f['ecg_r_height_std']=float(np.std(heights)) if len(heights) else np.nan
    freq,power=welch(c,fs=FS,nperseg=256);band=(freq>=.5)&(freq<=40);total=power[band].sum()
    f['ecg_dominant_frequency_hz']=float(freq[band][np.argmax(power[band])])
    f['ecg_low_power_fraction']=float(power[(freq>=.5)&(freq<5)].sum()/total) if total>0 else np.nan
    f['ecg_high_power_fraction']=float(power[(freq>=15)&(freq<=40)].sum()/total) if total>0 else np.nan
    pp=ppg_peaks[(ppg_peaks>=start)&(ppg_peaks<stop)]
    delays=[]
    for p in pp:
        i=np.searchsorted(rpeaks,p,side='right')-1
        if i>=0:
            delay=(p-rpeaks[i])/FS
            if .08<=delay<=.6:delays.append(delay)
    f['ecg_ppg_peak_delay_median_s']=float(np.median(delays)) if delays else np.nan
    f['ecg_ppg_peak_delay_iqr_s']=float(iqr(delays)) if delays else np.nan
    f['ecg_ppg_peak_match_fraction']=len(delays)/len(pp) if len(pp) else np.nan
    return f,len(rr),len(delays)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    import shutil
    for name in ['ppg_features.csv','training_folds.csv']:
        shutil.copy2(BASE/name,OUT/name)
    meta=pd.read_csv(BASE/'rows_and_labels.csv');assert set(meta.split)=={'train','validation'}
    original=pd.read_csv(BASE/'rows_and_labels.csv',low_memory=False).set_index('window_key')
    np.testing.assert_allclose(meta[['sbp','dbp']],original.loc[meta.window_key,['sbp','dbp']])
    # Establish the schema using a synthetic ECG; missing records use this schema.
    t=np.arange(1000)/FS
    example,_,_=ecg_features(np.sin(2*np.pi*t),np.sin(2*np.pi*t),np.arange(50,1000,125),np.arange(75,1000,125),125)
    names=list(example)
    rows={};audit=[]
    protocol=dict(date='2026-11-01',test_processed=False,cohort='Same 11,291 training / 857 validation windows as expanded RF',
        labels='Unchanged original complete-beat median ABP labels; ABP excluded from predictors',
        ECG='Full-record NeuroKit ECG cleaning and R-peak detection at 125Hz; window statistics, R timing, power and ECG-to-PPG-peak delay summaries',
        delay='Nearest preceding R peak, provisional 0.08–0.60s matching bounds; peak-to-peak delay is not validated pulse transit time',
        missing='Nonfinite/constant/failed ECG records retained with all ECG features missing; no zero replacement',
        fairness='PPG-only and PPG+ECG use identical window keys, labels, folds and scoring denominators',
        ecg_features=names,neurokit_version=nk.__version__)
    (OUT/'feature_protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs=list(meta.groupby(['part','record_key']))
    for number,((part,key),selected) in enumerate(jobs,1):
        rid=int(key.split(':')[-1]);status='ok';warning_text='';error=''
        with h5py.File(Path('data')/(part+'.mat'),'r') as h:
            record=h[h[part][rid,0]]
            raw=np.asarray(record[:,2],dtype=float);ppg=np.asarray(record[:,0],dtype=float)
        clean=np.full(len(raw),np.nan);r=np.array([],dtype=int);pp=np.array([],dtype=int)
        try:
            if not np.isfinite(raw).all() or np.ptp(raw)==0:raise ValueError('Nonfinite or constant ECG')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                clean=nk.ecg_clean(raw,sampling_rate=FS)
                _,info=nk.ecg_peaks(clean,sampling_rate=FS,correct_artifacts=False)
                r=np.asarray(info['ECG_R_Peaks'],dtype=int)
                ppg_clean=nk.ppg_clean(ppg,sampling_rate=FS)
                pp=np.asarray(nk.ppg_findpeaks(ppg_clean,sampling_rate=FS)['PPG_Peaks'],dtype=int)
            warning_text=' | '.join(sorted({str(w.message) for w in caught}))
            if warning_text:status='warning'
        except Exception as exc:status='failed';error=str(exc)
        for idx,s in selected.iterrows():
            start=int(s.window_key.split(':')[-1])*625
            expected=original.loc[s.window_key,'raw_ppg_sha256']
            assert hashlib.sha256(np.asarray(ppg[start:start+625],dtype='<f8').tobytes()).hexdigest()==expected
            f={n:np.nan for n in names};count=matched=0
            if status=='ok':
                f,count,matched=ecg_features(raw,clean,r,pp,start)
                if not np.isfinite(list(f.values())).any():status='failed'
            rows[idx]=f
            audit.append(dict(window_key=s.window_key,record_key=key,split=s.split,ecg_status=status,
                r_peaks=count,matched_ecg_ppg_peaks=matched,warning=warning_text,error=error))
        if number%100==0 or number==len(jobs):print('ECG records',number,'/',len(jobs),flush=True)
    ecg=pd.DataFrame.from_dict(rows,orient='index').sort_index()[names]
    ecg.replace([np.inf,-np.inf],np.nan).to_csv(OUT/'ecg_features.csv',index=False)
    meta.to_csv(OUT/'rows_and_labels.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'ecg_audit.csv',index=False)
    print(pd.DataFrame(audit).ecg_status.value_counts().to_string(),flush=True)
    (OUT/'FEATURES_READY.json').write_text(json.dumps(dict(rows=len(meta),ecg_features=len(names),test_processed=False)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    OUT=parser.parse_args().output_dir
    main()
