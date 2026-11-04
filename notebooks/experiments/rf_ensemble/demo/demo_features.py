"""Feature extraction used by the RF experiment. No pressure labels enter prediction."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
import neurokit2 as nk
from scipy import stats
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch
FS=125
WIDTH=625


def flat_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.where(edges == -1)[0] - np.where(edges == 1)[0], initial=0))

def rate_from_peaks(peaks):
    return float(60 / np.median(np.diff(peaks) / FS)) if len(peaks) >= 2 else np.nan

def describe_signal(x, prefix):
    return {prefix + 'mean': float(np.mean(x)), prefix + 'std': float(np.std(x)),
        prefix + 'iqr': float(stats.iqr(x)), prefix + 'skew': float(stats.skew(x)),
        prefix + 'kurtosis': float(stats.kurtosis(x)),
        prefix + 'p05': float(np.quantile(x, .05)), prefix + 'p95': float(np.quantile(x, .95))}

def extract_features(raw, clean, peaks):
    feats = describe_signal(raw, 'ppg_raw_')
    feats.update(describe_signal(clean, 'ppg_clean_'))
    vpg = np.gradient(clean, 1 / FS)
    apg = np.gradient(vpg, 1 / FS)
    feats.update(describe_signal(vpg, 'vpg_'))
    feats.update(describe_signal(apg, 'apg_'))
    gaps = np.diff(peaks) / FS
    feats.update(ppg_hr_bpm=rate_from_peaks(peaks),
        ppg_interval_std=float(np.std(gaps)) if len(gaps) else np.nan,
        ppg_interval_cv=float(np.std(gaps) / np.mean(gaps)) if len(gaps) else np.nan)
    amplitudes, widths, areas = [], [], []
    for left, right in zip(peaks[:-1], peaks[1:]):
        pulse = clean[left:right]
        low = pulse.min()
        amplitude = clean[left] - low
        amplitudes.append(amplitude)
        widths.append(float((pulse > low + amplitude / 2).sum() / FS))
        areas.append(float(np.trapezoid(pulse - low, dx=1 / FS)))
    feats.update(ppg_amplitude_mean=float(np.mean(amplitudes)) if amplitudes else np.nan,
        ppg_amplitude_std=float(np.std(amplitudes)) if amplitudes else np.nan,
        ppg_half_height_duration=float(np.mean(widths)) if widths else np.nan,
        ppg_baseline_subtracted_area=float(np.mean(areas)) if areas else np.nan)
    frequency, power = welch(clean, fs=FS, nperseg=256)
    band = (frequency >= .5) & (frequency <= 5)
    band_power = power[band].sum()
    feats['ppg_dominant_frequency_hz'] = float(frequency[band][np.argmax(power[band])])
    feats['ppg_high_frequency_fraction'] = float(power[(frequency >= 2) & (frequency <= 5)].sum() / band_power) if band_power > 0 else np.nan
    return feats

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
