"""Run a saved research RF on one window from a 125 Hz PPG recording.

The input is a 1-D .npy recording, not an ABP signal or a precomputed feature row.
Keep recording context around the requested five-second window, as in training.
No clinical-use claim or pressure-reference quality check is made at inference.
"""
from pathlib import Path
import argparse,json
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from expanded_data_analysis import extract_features,flat_run
from preprocessing_ablation import pulse_features

FS=125
WIDTH=625


def window_features(ppg,start):
    ppg=np.asarray(ppg,dtype=float)
    if ppg.ndim!=1 or not np.isfinite(ppg).all() or np.ptp(ppg)==0:
        raise ValueError('PPG must be a finite, nonconstant one-dimensional recording')
    if not isinstance(start,(int,np.integer)) or start<FS or start+WIDTH>len(ppg)-FS:
        raise ValueError('Provide a full five-second window with at least one second of recording context on each side')
    signal,info=nk.ppg_process(ppg,sampling_rate=FS)
    clean=signal.PPG_Clean.to_numpy()
    peaks=np.asarray(info['PPG_Peaks'],dtype=int)
    local=peaks[(peaks>=start)&(peaks<start+WIDTH)]-start
    quality=float(signal.PPG_Quality.iloc[start:start+WIDTH].mean())
    if len(local)<3 or not np.isfinite(quality) or quality<.8:
        raise ValueError('PPG fails the training pulse-count or template-quality threshold')
    raw=ppg[start:start+WIDTH];filtered=clean[start:start+WIDTH]
    if max(flat_run(raw==raw.min()),flat_run(raw==raw.max()))>=5:
        raise ValueError('PPG has a flat extreme excluded by the training rules')
    features=extract_features(raw,filtered,local)
    shape,count=pulse_features(filtered,local);features.update(shape)
    return features,dict(ppg_template_quality=quality,ppg_peaks=int(len(local)),complete_pulses=count)


def predict(ppg,start,model_path):
    model=joblib.load(model_path)
    names=list(model.feature_names_in_)
    if any(not name.startswith(('ppg_','vpg_','apg_','pulse_')) for name in names):
        raise ValueError('Model has a predictor outside the PPG-only whitelist')
    features,audit=window_features(ppg,start)
    if not set(names)<=set(features):raise ValueError('Model requires an unsupported feature')
    frame=pd.DataFrame([features]).reindex(columns=names).replace([np.inf,-np.inf],np.nan)
    prediction=model.predict(frame)[0]
    if prediction.shape!=(2,) or not np.isfinite(prediction).all():raise ValueError('Invalid model output')
    return dict(sbp_mmHg=float(prediction[0]),dbp_mmHg=float(prediction[1]),
        model=str(model_path),sampling_rate_hz=FS,start_sample=int(start),**audit,
        status='Research prediction; ABP reference checks unavailable at inference')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ppg-npy',type=Path,required=True)
    parser.add_argument('--start-sample',type=int,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(predict(np.load(args.ppg_npy,allow_pickle=False),args.start_sample,args.model),indent=2))
