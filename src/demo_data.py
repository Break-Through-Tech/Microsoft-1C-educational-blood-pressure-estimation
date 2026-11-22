"""Seeded synthetic PPG windows for demo mode (no dataset or artifacts needed).

The waveforms are generated, not measured, and the labels are invented from a made-up rule, so
nothing computed on them says anything about real model accuracy. Everything is flagged "demo".
"""
from __future__ import annotations

import numpy as np

from .preprocess import TARGET_FS, WINDOW_SIZE, clean_window

DEMO_SEED = 20261120
N_DEMO_SAMPLES = 50
DEMO_SECONDS = 20          # generate a longer signal, then clean it like a real recording


def _synthetic_ppg(rng: np.random.Generator, heart_rate: float, seconds: float, fs: int = TARGET_FS) -> np.ndarray:
    """Pulse train: systolic peak + smaller dicrotic wave per beat, beat-to-beat jitter,
    baseline wander and noise."""
    t = np.arange(int(seconds * fs)) / fs
    beat = 60 / heart_rate
    x = np.zeros_like(t)
    onset = rng.uniform(0, beat)
    while onset < seconds:
        x += np.exp(-0.5 * ((t - onset - 0.15 * beat) / (0.07 * beat)) ** 2)
        x += 0.35 * np.exp(-0.5 * ((t - onset - 0.45 * beat) / (0.09 * beat)) ** 2)
        onset += beat * rng.normal(1, 0.03)
    x += 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0, 0.02, len(t))
    return 2.0 + x


def make_demo_samples(n: int = N_DEMO_SAMPLES, seed: int = DEMO_SEED) -> dict:
    """n synthetic windows in the standard input format plus invented labels.

    Returns X (n, 1, 625) float32, y (n, 2) float32 [SBP, DBP] mmHg, subject_ids (n,) int64
    (10 fake subjects), raw_ppg (n, 625) float32, heart_rate (n,).
    """
    rng = np.random.default_rng(seed)
    n_subjects = 10
    subject_ids = np.repeat(np.arange(n_subjects), int(np.ceil(n / n_subjects)))[:n]
    subject_hr = rng.uniform(55, 105, n_subjects)
    subject_bp = np.column_stack([rng.uniform(100, 150, n_subjects), rng.uniform(55, 90, n_subjects)])

    X, y, raw, hrs = [], [], [], []
    for sid in subject_ids:
        k = sid
        hr = subject_hr[k] + rng.normal(0, 3)
        sig = _synthetic_ppg(rng, hr, DEMO_SECONDS)
        mid = (len(sig) - WINDOW_SIZE) // 2
        X.append(clean_window(sig))                               # filtered over the full 20 s
        raw.append(sig[mid:mid + WINDOW_SIZE])
        sbp = subject_bp[k, 0] + 0.2 * (hr - 75) + rng.normal(0, 4)
        dbp = min(subject_bp[k, 1] + 0.1 * (hr - 75) + rng.normal(0, 3), sbp - 20)
        y.append((sbp, dbp))
        hrs.append(hr)
    return {"X": np.stack(X).astype(np.float32), "y": np.array(y, dtype=np.float32),
            "subject_ids": subject_ids.astype(np.int64), "raw_ppg": np.array(raw, dtype=np.float32),
            "heart_rate": np.array(hrs, dtype=np.float32), "source": "synthetic (demo)"}
