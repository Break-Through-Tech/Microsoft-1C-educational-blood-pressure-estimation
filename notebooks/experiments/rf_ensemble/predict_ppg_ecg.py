"""Research prediction from synchronized 125 Hz PPG and ECG recordings.

Supports a saved two-output feature model from compare_ppg_ecg_models.py.
PPG and ECG must have the same sample count and time origin. No ABP is used.
"""
from pathlib import Path
import argparse,json,warnings
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from predict_ppg_rf import window_features
from prepare_ecg_comparison import ecg_features


def predict(ppg,ecg,start,model_path):
    ppg=np.asarray(ppg,dtype=float);ecg=np.asarray(ecg,dtype=float)
    if ecg.ndim!=1 or ppg.ndim!=1 or len(ecg)!=len(ppg):raise ValueError('Supply synchronized one-dimensional PPG/ECG recordings of equal length')
    if not np.isfinite(ecg).all() or np.ptp(ecg)==0:raise ValueError('ECG must be finite and nonconstant')
    f,audit=window_features(ppg,start)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        clean=nk.ecg_clean(ecg,sampling_rate=125)
        _,info=nk.ecg_peaks(clean,sampling_rate=125,correct_artifacts=False)
        r=np.asarray(info['ECG_R_Peaks'],dtype=int)
        ppg_clean=nk.ppg_clean(ppg,sampling_rate=125)
        pp=np.asarray(nk.ppg_findpeaks(ppg_clean,sampling_rate=125)['PPG_Peaks'],dtype=int)
    if captured:raise ValueError('Signal processing produced warnings; inspect the recording before prediction')
    extra,count,matched=ecg_features(ecg,clean,r,pp,start);f.update(extra)
    model=joblib.load(model_path);names=list(model.feature_names_in_)
    if not any(n.startswith('ecg_') for n in names):raise ValueError('Use an explicitly trained PPG+ECG model')
    if any(not n.startswith(('ppg_','vpg_','apg_','pulse_','ecg_')) for n in names):raise ValueError('Unsupported predictor in model')
    X=pd.DataFrame([f]).reindex(columns=names).replace([np.inf,-np.inf],np.nan)
    y=model.predict(X)[0]
    if np.shape(y)!=(2,) or not np.isfinite(y).all():raise ValueError('Expected finite SBP/DBP outputs')
    return dict(sbp_mmHg=float(y[0]),dbp_mmHg=float(y[1]),input_modality='PPG + ECG',start_sample=int(start),
        ecg_r_peaks=count,matched_ecg_ppg_peaks=matched,**audit,
        status='Research prediction; synchronized inputs assumed; no ABP reference checks')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppg-npy',type=Path,required=True);p.add_argument('--ecg-npy',type=Path,required=True)
    p.add_argument('--start-sample',type=int,required=True);p.add_argument('--model',type=Path,required=True)
    a=p.parse_args();print(json.dumps(predict(np.load(a.ppg_npy,allow_pickle=False),np.load(a.ecg_npy,allow_pickle=False),a.start_sample,a.model),indent=2))
