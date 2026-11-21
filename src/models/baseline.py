"""Naive baseline: always predict the training-set mean SBP and DBP. Every metric is reported next to it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import BPModel, check_output


class MeanBaseline(BPModel):
    name = "mean_baseline"
    display_name = "Mean baseline"
    input_kind = "waveform"        # ignores the input; any shape with len(X) rows works
    is_baseline = True

    def __init__(self, means=None):
        self.means = None if means is None else np.asarray(means, dtype=np.float32)

    def fit(self, X, y: np.ndarray) -> "MeanBaseline":
        self.means = np.asarray(y, dtype=np.float64).mean(axis=0).astype(np.float32)
        return self

    def predict(self, X) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("MeanBaseline is not fitted")
        return check_output(np.broadcast_to(self.means, (len(X), 2)).copy(), len(X))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sbp_mean": float(self.means[0]), "dbp_mean": float(self.means[1]),
                                    "fitted_on": "train split"}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "MeanBaseline":
        d = json.loads(Path(path).read_text())
        return cls([d["sbp_mean"], d["dbp_mean"]])
