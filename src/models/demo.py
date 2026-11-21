"""Placeholder models for demo mode, used only when the real artifacts are missing.

They exist so the app runs end to end without data; their numbers are meaningless and are always
served with mode "demo".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from ..preprocess import TARGET_FS, WINDOW_SIZE
from .base import BPModel, check_output
from .baseline import MeanBaseline


class DemoPulseModel(BPModel):
    """Made-up rule: BP rises slightly with pulse rate. Not trained on anything."""

    name = "demo_placeholder"
    display_name = "Placeholder model (demo)"
    input_kind = "waveform"
    is_demo = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        hr = np.array([self._heart_rate(x) for x in X])
        sbp = 118 + 0.25 * (hr - 75)
        dbp = 74 + 0.12 * (hr - 75)
        return check_output(np.column_stack([sbp, dbp]), len(X))

    @staticmethod
    def _heart_rate(x: np.ndarray) -> float:
        peaks, _ = find_peaks(x, distance=int(0.33 * TARGET_FS), prominence=0.2)
        return 60 / np.median(np.diff(peaks)) * TARGET_FS if len(peaks) > 1 else 75.0

    def explain(self, X: np.ndarray, n_segments: int = 10) -> dict:
        n = len(X)
        bounds = np.array_split(np.arange(WINDOW_SIZE), n_segments)
        flat = np.full((n, n_segments), 1 / n_segments).round(4).tolist()
        return {"method": "uniform (demo placeholder)",
                "segments": [{"start_s": b[0] / TARGET_FS, "end_s": (b[-1] + 1) / TARGET_FS} for b in bounds],
                "sbp": flat, "dbp": flat}

    @classmethod
    def load(cls, path: Path | None = None) -> "DemoPulseModel":
        return cls()


class DemoMeanBaseline(MeanBaseline):
    name = "mean_baseline"
    display_name = "Mean baseline (demo)"
    is_demo = True


def demo_models(y_train_like: np.ndarray) -> dict[str, BPModel]:
    baseline = DemoMeanBaseline().fit(None, y_train_like)
    return {DemoPulseModel.name: DemoPulseModel(), baseline.name: baseline}
