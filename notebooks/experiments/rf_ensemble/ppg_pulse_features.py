"""Complete-pulse measurements used by the RF feature pipeline."""
import numpy as np
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
