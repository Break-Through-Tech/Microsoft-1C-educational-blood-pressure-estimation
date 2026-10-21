"""
Windowing + feature extraction shared by the modeling notebooks.

The feature functions are the ones developed in data_exploration_pyscript,
moved here so model scripts can import them without re-running the EDA.
"""

import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats as sstats
from scipy.signal import find_peaks

FS = 125           # sampling rate (Hz)
WINDOW_SIZE = 625  # 5 seconds at 125Hz


# ---------------------------------------------------------------------------
# Loading + windowing
# ---------------------------------------------------------------------------

def iter_windows(mat_path, part_key, windows_per_record=None, rng=None, max_records=None):
    """
    Yield (record_id, window_idx, ppg_window, abp_window) from one Part_X.mat file.

    Rows of each record: 0 = PPG, 1 = ABP, 2 = ECG.
    windows_per_record caps how many 5s windows we take from each recording
    (randomly chosen) so one long recording can't dominate the dataset.
    max_records limits how many recordings are read (handy for quick test runs).
    """
    with h5py.File(mat_path, 'r') as f:
        refs = f[part_key]
        n_records = refs.shape[0] if max_records is None else min(max_records, refs.shape[0])
        for rec_id in range(n_records):
            mat = np.array(f[refs[rec_id, 0]]).T
            n_windows = mat.shape[1] // WINDOW_SIZE
            idxs = np.arange(n_windows)
            if windows_per_record is not None and n_windows > windows_per_record:
                idxs = np.sort(rng.choice(idxs, windows_per_record, replace=False))
            for j in idxs:
                start, end = j * WINDOW_SIZE, (j + 1) * WINDOW_SIZE
                yield rec_id, j, mat[0, start:end], mat[1, start:end]


def bp_labels(abp, fs=FS):
    """SBP = median of ABP peaks, DBP = median of ABP valleys (more robust than max/min)."""
    sys_idx, _ = find_peaks(abp, distance=int(0.4 * fs))
    dia_idx, _ = find_peaks(-abp, distance=int(0.4 * fs))
    if len(sys_idx) < 2 or len(dia_idx) < 2:
        return None, None
    return np.median(abp[sys_idx]), np.median(abp[dia_idx])


def is_plausible(sbp, dbp):
    """Same physiological range filter used in the EDA."""
    if sbp is None or dbp is None:
        return False
    pulse_pressure = sbp - dbp
    return (60 <= sbp <= 220) and (30 <= dbp <= 130) and pulse_pressure >= 10


# ---------------------------------------------------------------------------
# Feature functions (from data_exploration_pyscript)
# ---------------------------------------------------------------------------

def extract_ppg_features(ppg_signals, ppg_info, sampling_rate=FS):
    """PPG window information from output of nk.ppg_process"""
    peaks = ppg_info['PPG_Peaks']
    clean = ppg_signals['PPG_Clean'].values

    if len(peaks) < 2:
        return None

    peak_intervals = np.diff(peaks) / sampling_rate

    amplitudes, widths, areas = [], [], []
    for i in range(len(peaks) - 1):
        pulse = clean[peaks[i]:peaks[i + 1]]
        trough = pulse.min()
        amplitudes.append(clean[peaks[i]] - trough)
        half_max = (clean[peaks[i]] + trough) / 2
        widths.append((pulse > half_max).sum() / sampling_rate)
        areas.append(np.trapezoid(pulse) / sampling_rate)

    vpg = np.gradient(clean)
    apg = np.gradient(vpg)

    return {
        'ppg_hr_bpm': 60 / np.mean(peak_intervals),
        'ppg_interval_std': np.std(peak_intervals),
        'ppg_amplitude_mean': np.mean(amplitudes),
        'ppg_amplitude_std': np.std(amplitudes),
        'ppg_pulse_width_mean': np.mean(widths),
        'ppg_pulse_area_mean': np.mean(areas),
        'ppg_apg_ba_ratio': apg.min() / apg.max() if apg.max() != 0 else np.nan,
    }


def extract_ppg_fiducial_features(ppg_signals, ppg_info, sampling_rate=FS):
    """Systolic peak / dicrotic notch / diastolic peak ratios and timings."""
    clean = ppg_signals['PPG_Clean'].values
    peaks = ppg_info['PPG_Peaks']
    vpg = np.gradient(clean)

    if len(peaks) < 3:
        return None

    dia_sys_ratios, notch_sys_ratios = [], []
    t_foot_sys, t_foot_notch, t_foot_dia, t_sys_dia = [], [], [], []

    for i in range(1, len(peaks) - 1):
        foot = peaks[i - 1] + np.argmin(clean[peaks[i - 1]:peaks[i] + 1])
        next_foot = peaks[i] + np.argmin(clean[peaks[i]:peaks[i + 1] + 1])

        vseg = vpg[peaks[i]:next_foot + 1]
        up = nk.signal_zerocrossings(vseg, direction='up')
        if len(up) == 0:
            continue
        notch_rel = up[0]

        down = nk.signal_zerocrossings(vseg[notch_rel:], direction='down')
        if len(down) == 0:
            continue
        dia_rel = notch_rel + down[0]

        notch_idx = peaks[i] + notch_rel
        dia_idx = peaks[i] + dia_rel

        baseline = clean[foot]
        sys_amp = clean[peaks[i]] - baseline
        if sys_amp <= 0:
            continue

        dia_sys_ratios.append((clean[dia_idx] - baseline) / sys_amp)
        notch_sys_ratios.append((clean[notch_idx] - baseline) / sys_amp)
        t_foot_sys.append((peaks[i] - foot) / sampling_rate)
        t_foot_notch.append((notch_idx - foot) / sampling_rate)
        t_foot_dia.append((dia_idx - foot) / sampling_rate)
        t_sys_dia.append((dia_idx - peaks[i]) / sampling_rate)

    if len(dia_sys_ratios) == 0:
        return None

    return {
        'ppg_dia_sys_ratio': np.mean(dia_sys_ratios),
        'ppg_notch_sys_ratio': np.mean(notch_sys_ratios),
        't_foot_to_sys': np.mean(t_foot_sys),
        't_foot_to_notch': np.mean(t_foot_notch),
        't_foot_to_dia': np.mean(t_foot_dia),
        't_sys_to_dia': np.mean(t_sys_dia),
        'n_pulses_with_notch': len(dia_sys_ratios),
    }


def extract_window_statistics(signal, sampling_rate=FS, prefix=''):
    """Non-fiducial statistics (PPG, VPG, or APG)."""
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)

    shannon_entropy, _ = nk.entropy_shannon(signal, symbolize=10)

    tkeo = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    crossings = nk.signal_zerocrossings(centered)
    zcr = len(crossings) / (len(signal) / sampling_rate)

    return {
        f'{prefix}mean': np.mean(signal),
        f'{prefix}median': np.median(signal),
        f'{prefix}std': np.std(signal),
        f'{prefix}var': np.var(signal),
        f'{prefix}iqr': sstats.iqr(signal),
        f'{prefix}skew': sstats.skew(signal),
        f'{prefix}kurtosis': sstats.kurtosis(signal),
        f'{prefix}shannon_entropy': shannon_entropy,
        f'{prefix}tkeo_var': np.var(tkeo),
        f'{prefix}zcr': zcr,
    }


def window_features(ppg_window, fs=FS):
    """All features for one 5s PPG window, or None if it can't be processed."""
    try:
        ppg_sig, ppg_info = nk.ppg_process(ppg_window, sampling_rate=fs)
        feats = extract_ppg_features(ppg_sig, ppg_info, fs)
        if feats is None:
            return None

        fiducial = extract_ppg_fiducial_features(ppg_sig, ppg_info, fs)
        if fiducial is not None:
            feats.update(fiducial)

        clean = ppg_sig['PPG_Clean'].values
        vpg = np.gradient(clean)
        apg = np.gradient(vpg)
        feats.update(extract_window_statistics(clean, fs, prefix='ppg_stat_'))
        feats.update(extract_window_statistics(vpg, fs, prefix='vpg_stat_'))
        feats.update(extract_window_statistics(apg, fs, prefix='apg_stat_'))
        return feats
    except Exception:
        return None


def _process_window(rec_id, win_idx, ppg_window, abp_window):
    sbp, dbp = bp_labels(abp_window)
    if not is_plausible(sbp, dbp):
        return None
    feats = window_features(ppg_window)
    if feats is None:
        return None
    return {'record': rec_id, 'window': win_idx, 'sbp': sbp, 'dbp': dbp, **feats}


def build_feature_table(mat_path, part_key, windows_per_record=10, max_records=None,
                        n_jobs=1, seed=42):
    """
    Window one Part_X.mat file, label each window from ABP, drop implausible
    windows, and extract PPG features. Returns one row per window.

    n_jobs > 1 runs NeuroKit in parallel (-1 = all cores).
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    windows = iter_windows(mat_path, part_key, windows_per_record, rng, max_records)
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(_process_window)(*w) for w in windows
    )
    return pd.DataFrame([r for r in rows if r is not None])
