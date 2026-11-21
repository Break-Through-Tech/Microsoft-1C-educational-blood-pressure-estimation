"""Data loading and preprocessing, reproducing the ResNet notebook (Claire, model_training.ipynb, v3 settings).

Standard input for every model (see CLAUDE.md):
    PPG only, TARGET_FS = 125 Hz, WINDOW_SIZE = 625 samples (5 s), shape (n, 1, 625) float32.
    1. mean removal and 2. 4th-order Butterworth band-pass 0.5-5 Hz (zero phase) on the WHOLE signal,
    3. cut non-overlapping 5 s windows, 4. divide each window by its max |value| -> [-1, 1].

"Subject ID" = UCI recording ID. The dataset has no patient IDs, so one person may own several
recordings; the split is by recording, the closest available proxy.
Recording ID = (part - 1) * 3000 + record index (0-based), i.e. the notebook's rec_counter.
"""
from __future__ import annotations

import json
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from scipy.stats import kurtosis, skew

from .paths import Paths

TARGET_FS = 125
WINDOW_SECONDS = 5
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS      # 625
BANDPASS_HZ = (0.5, 5.0)
FILTER_ORDER = 4
MIN_SECONDS = 2.0                             # clean_window rejects shorter input
RECORDS_PER_PART = 3000
N_PARTS = 4

# bump when anything that changes the cached windows changes
CACHE_VERSION = "ppg-v3-1"


# ---------------------------------------------------------------------------
# Signal steps (same calls and parameters as the notebook)
# ---------------------------------------------------------------------------

# These two reproduce the notebook's neurokit2 calls exactly (bit-identical output, checked in
# tests/test_preprocess_equivalence.py) without importing neurokit2 at runtime, which saves ~60 MB of
# memory and its matplotlib/scikit-learn dependencies in the deployed app.

def remove_mean(x: np.ndarray) -> np.ndarray:
    """= nk.signal_detrend(x, order=0): subtract a degree-0 polynomial fit on linspace(0, 100, n)."""
    x = np.asarray(x, dtype=float)
    grid = np.linspace(0, 100, len(x))
    return x - np.polyval(np.polyfit(grid, x, 0), grid)


@lru_cache(maxsize=8)
def _butter_sos(order: int, low: float, high: float, fs: float) -> np.ndarray:
    return butter(order, [low, high], btype="bandpass", output="sos", fs=fs)


def bandpass_filter(x: np.ndarray, fs: int = TARGET_FS, low: float = BANDPASS_HZ[0],
                    high: float = BANDPASS_HZ[1], order: int = FILTER_ORDER) -> np.ndarray:
    """= nk.signal_filter(x, method="butterworth"): Butterworth SOS band-pass, zero-phase (sosfiltfilt)."""
    return sosfiltfilt(_butter_sos(order, low, high, float(fs)), np.asarray(x, dtype=float))


def normalize(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


def resample_to_target(x: np.ndarray, fs: float) -> np.ndarray:
    """Polyphase resampling from fs to TARGET_FS (no-op at 125 Hz)."""
    x = np.asarray(x, dtype=float)
    if fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {fs}")
    if abs(fs - TARGET_FS) < 1e-9:
        return x
    ratio = Fraction(TARGET_FS / fs).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def filter_signal(x: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """Resample to TARGET_FS, then mean removal + band-pass over the whole signal."""
    x = resample_to_target(x, fs)
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or infinite values")
    return bandpass_filter(remove_mean(x))


def fix_length(x: np.ndarray, n: int = WINDOW_SIZE) -> np.ndarray:
    """Centre-crop longer input; zero-pad shorter input at the end (the filtered signal is zero-mean)."""
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    return np.pad(x, (0, n - len(x)))


def clean_window(x: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """One PPG segment in any sampling rate -> model input of shape (1, 625), float32.

    Resample -> mean removal + band-pass on the whole segment -> fix length -> max-abs normalize.
    Training filtered the full recording before windowing, so pass the longest signal you have:
    filtering a bare 5 s clip adds edge effects the model never saw.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) / fs < MIN_SECONDS:
        raise ValueError(f"need at least {MIN_SECONDS:g} s of signal, got {len(x) / fs:.2f} s")
    filtered = filter_signal(x, fs)
    return normalize(fix_length(filtered)).astype(np.float32)[None, :]


def windows_from_signal(x: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """A long PPG recording -> all non-overlapping windows, shape (k, 1, 625), exactly like training."""
    filtered = filter_signal(x, fs)
    k = len(filtered) // WINDOW_SIZE
    if k == 0:
        raise ValueError(f"need at least {WINDOW_SECONDS} s of signal after resampling")
    wins = [normalize(filtered[j * WINDOW_SIZE:(j + 1) * WINDOW_SIZE]) for j in range(k)]
    return np.stack(wins).astype(np.float32)[:, None, :]


# ---------------------------------------------------------------------------
# Labels and quality filter (notebook v3: ALIGN=False, STRICT_QC=False)
# ---------------------------------------------------------------------------

def get_bp_labels(abp_window: np.ndarray, fs: int = TARGET_FS):
    """(SBP, DBP) in mmHg from one ABP window: median of beat peaks / valleys, or None if unusable."""
    min_gap = fs * 0.4
    peaks, _ = find_peaks(abp_window, distance=min_gap)
    valleys, _ = find_peaks(-abp_window, distance=min_gap)
    if len(peaks) == 0 or len(valleys) == 0:
        return None
    sys_vals = abp_window[peaks]
    sys_vals = sys_vals[(sys_vals >= 70) & (sys_vals <= 220)]
    dia_vals = abp_window[valleys]
    dia_vals = dia_vals[(dia_vals >= 30) & (dia_vals <= 140)]
    if len(sys_vals) == 0 or len(dia_vals) == 0:
        return None
    sbp, dbp = np.median(sys_vals), np.median(dia_vals)
    if sbp <= dbp or sbp - dbp < 10:
        return None
    return sbp, dbp


def good_ppg_window(ppg_win: np.ndarray, fs: int = TARGET_FS, min_hr: int = 48, max_hr: int = 180) -> bool:
    """Reject flat / non-finite windows and windows with an implausible beat count."""
    if not np.isfinite(ppg_win).all() or np.std(ppg_win) < 1e-3:
        return False
    peaks, _ = find_peaks(ppg_win, distance=int(0.33 * fs), prominence=0.2)
    n_sec = len(ppg_win) / fs
    return min_hr * n_sec / 60 <= len(peaks) <= max_hr * n_sec / 60


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _cache_files(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / f"windows_{CACHE_VERSION}.npy", cache_dir / f"windows_{CACHE_VERSION}_meta.npz"


def _build_cache(raw_dir: Path, cache_dir: Path, verbose: bool = True) -> None:
    """Window every recording like the notebook and write X to disk part by part (keeps RAM low)."""
    import h5py

    missing = [f"Part_{p}.mat" for p in range(1, N_PARTS + 1) if not (raw_dir / f"Part_{p}.mat").exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {raw_dir}; download the UCI Cuff-Less BP Estimation "
                                "dataset (https://archive.ics.uci.edu/dataset/340) and unzip Part_*.mat there")
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, meta_path = _cache_files(cache_dir)

    part_files, ys, subjects, parts, records, starts = [], [], [], [], [], []
    n_total = 0
    t0 = time.time()
    for part in range(1, N_PARTS + 1):
        xs = []
        with h5py.File(raw_dir / f"Part_{part}.mat", "r") as f:
            data = f[f"Part_{part}"]
            if data.shape[0] != RECORDS_PER_PART:
                raise ValueError(f"Part_{part} has {data.shape[0]} records, expected {RECORDS_PER_PART}")
            for i in range(data.shape[0]):
                rec_id = (part - 1) * RECORDS_PER_PART + i       # == notebook rec_counter
                record = np.array(f[data[i, 0]]).T                # (3, n): PPG, ABP, ECG
                ppg, abp = record[0], record[1]
                if not (np.isfinite(ppg).all() and np.isfinite(abp).all()):
                    continue
                filtered = bandpass_filter(remove_mean(ppg))
                for j in range(len(filtered) // WINDOW_SIZE):
                    s, e = j * WINDOW_SIZE, (j + 1) * WINDOW_SIZE
                    n_total += 1
                    win = normalize(filtered[s:e])
                    labels = get_bp_labels(abp[s:e])
                    if labels is None or not good_ppg_window(win):
                        continue
                    xs.append(win.astype(np.float32))
                    ys.append(labels)
                    subjects.append(rec_id)
                    parts.append(part)
                    records.append(i)
                    starts.append(s)
        part_file = cache_dir / f"_part{part}.npy"
        np.save(part_file, np.stack(xs)[:, None, :])
        part_files.append(part_file)
        del xs
        if verbose:
            print(f"  Part_{part}: {len(ys):,} windows kept so far ({time.time() - t0:.0f} s)")

    n = len(ys)
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, 1, WINDOW_SIZE))
    pos = 0
    for pf in part_files:
        chunk = np.load(pf, mmap_mode="r")
        X[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
        del chunk
        pf.unlink()
    X.flush()
    del X
    np.savez(meta_path, y=np.array(ys, dtype=np.float32), subject_ids=np.array(subjects, dtype=np.int64),
             part=np.array(parts, dtype=np.int8), record=np.array(records, dtype=np.int16),
             start=np.array(starts, dtype=np.int32), n_total_windows=n_total,
             info=json.dumps({"cache_version": CACHE_VERSION, "fs": TARGET_FS, "window": WINDOW_SIZE,
                              "subject_id": "UCI recording id = (part-1)*3000 + record index"}))
    if verbose:
        print(f"  kept {n:,} / {n_total:,} windows ({n / n_total:.1%}) from "
              f"{len(np.unique(subjects)):,} recordings -> {x_path}")


def load_dataset(paths: Paths | None = None, rebuild: bool = False, return_meta: bool = False,
                 verbose: bool = True):
    """All labelled windows of the UCI dataset, preprocessed exactly like the ResNet notebook (v3).

    Returns X (N, 1, 625) float32 (read-only memory map), subject_ids (N,) int64 recording IDs,
    y (N, 2) float32 [SBP, DBP] in mmHg; with return_meta=True also a dict with part, record and
    start sample of every window. The first call builds a disk cache in data/cache/ (several minutes).
    """
    paths = paths or Paths()
    x_path, meta_path = _cache_files(paths.cache_dir)
    if rebuild or not (x_path.exists() and meta_path.exists()):
        if verbose:
            print(f"Building window cache from {paths.raw_dir} ...")
        _build_cache(paths.raw_dir, paths.cache_dir, verbose=verbose)
    X = np.load(x_path, mmap_mode="r")
    with np.load(meta_path) as m:
        y, subject_ids = m["y"], m["subject_ids"]
        meta = {"part": m["part"], "record": m["record"], "start": m["start"],
                "n_total_windows": int(m["n_total_windows"])}
    if len(X) != len(y):
        raise RuntimeError(f"cache is inconsistent ({len(X)} windows vs {len(y)} labels); rerun with rebuild=True")
    return (X, subject_ids, y, meta) if return_meta else (X, subject_ids, y)


def read_raw_window(raw_dir: Path, part: int, record: int, start: int, n: int = WINDOW_SIZE) -> dict:
    """Unprocessed PPG and ABP for one window, for display (ABP is reference only, never a model input)."""
    import h5py
    with h5py.File(Path(raw_dir) / f"Part_{part}.mat", "r") as f:
        rec = np.array(f[f[f"Part_{part}"][record, 0]]).T
    return {"ppg": rec[0, start:start + n].astype(np.float32), "abp": rec[1, start:start + n].astype(np.float32)}


# ---------------------------------------------------------------------------
# Features (stub for feature-based models added later)
# ---------------------------------------------------------------------------

FEATURE_NAMES = ("heart_rate_bpm", "n_peaks", "peak_amplitude_mean", "peak_amplitude_std",
                 "ibi_std_s", "signal_std", "skewness", "kurtosis")


def extract_features(window: np.ndarray, fs: int = TARGET_FS) -> dict:
    """Simple per-window PPG statistics from a cleaned window (shape (625,) or (1, 625)).

    A stub: enough for a first feature-based model, not a replacement for teammates' feature sets.
    """
    w = np.asarray(window, dtype=float).ravel()
    peaks, _ = find_peaks(w, distance=int(0.33 * fs), prominence=0.2)
    ibi = np.diff(peaks) / fs
    return {
        "heart_rate_bpm": float(60 / np.median(ibi)) if len(ibi) else float("nan"),
        "n_peaks": float(len(peaks)),
        "peak_amplitude_mean": float(w[peaks].mean()) if len(peaks) else float("nan"),
        "peak_amplitude_std": float(w[peaks].std()) if len(peaks) else float("nan"),
        "ibi_std_s": float(ibi.std()) if len(ibi) > 1 else float("nan"),
        "signal_std": float(w.std()),
        "skewness": float(skew(w)),
        "kurtosis": float(kurtosis(w)),
    }


def features_matrix(X: np.ndarray) -> np.ndarray:
    """(n, 1, 625) windows -> (n, len(FEATURE_NAMES)) float array, columns in FEATURE_NAMES order."""
    return np.array([[extract_features(x)[k] for k in FEATURE_NAMES] for x in X], dtype=float)
