import h5py
import numpy as np

with h5py.File("data/Part_1.mat", "r") as f:
    data = f["Part_1"]

    # Get the first record from the MATLAB cell array
    ref = data[0, 0]
    record = np.array(f[ref])

    print("Shape:", record.shape)
    print(record[:5])
    


import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks

FS = 125  # sample rate

def extract_features(record, fs=FS):
    ppg, abp, ecg = record  

    ppg_signals, ppg_info = nk.ppg_process(ppg, sampling_rate=fs)
    ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=fs)
    ppg_peaks = np.asarray(ppg_info["PPG_Peaks"])
    r_peaks = np.asarray(ecg_info["ECG_R_Peaks"])

    feats = {
        "ppg_rate_mean": ppg_signals["PPG_Rate"].mean(),
        "ecg_rate_mean": ecg_signals["ECG_Rate"].mean(),
        "ppg_peak_amp_mean": ppg_signals["PPG_Clean"].values[ppg_peaks].mean(),
    }

    # HRV from PPG peaks
    feats.update(nk.hrv_time(ppg_peaks, sampling_rate=fs).iloc[0].to_dict())

    # Pulse transit time
    ptt = []
    for r in r_peaks:
        nxt = ppg_peaks[(ppg_peaks > r) & (ppg_peaks < r + int(0.6 * fs))]
        if len(nxt):
            ptt.append((nxt[0] - r) / fs)
    feats["ptt_mean"] = np.mean(ptt) if ptt else np.nan
    feats["ptt_std"] = np.std(ptt) if ptt else np.nan

    # Target features
    sys_idx, _ = find_peaks(abp, distance=int(0.4 * fs))
    dia_idx, _ = find_peaks(-abp, distance=int(0.4 * fs))
    feats["SBP"] = np.median(abp[sys_idx])
    feats["DBP"] = np.median(abp[dia_idx])
    return feats