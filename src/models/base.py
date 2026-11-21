"""The interface every blood-pressure model implements (see CLAUDE.md)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

InputKind = Literal["waveform", "features"]


class BPModel(ABC):
    """A model that maps standard inputs to [SBP, DBP] in mmHg.

    input_kind "waveform": predict() gets cleaned windows, shape (n, 1, 625) (see preprocess.py).
    input_kind "features": predict() gets preprocess.features_matrix(windows), shape (n, n_features).

    Optional: a model with fit(X, y) and save(path) is trained on the train split by
    scripts/evaluate.py before it is evaluated.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    input_kind: ClassVar[InputKind] = "waveform"
    is_baseline: ClassVar[bool] = False      # baselines are reported but never chosen as primary
    is_demo: ClassVar[bool] = False          # placeholder models for demo mode

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions in mmHg, shape (n, 2), columns [SBP, DBP]."""

    def explain(self, X: np.ndarray) -> dict:
        """What drove the prediction. Models without an explanation return method "none"."""
        return {"method": "none"}

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BPModel":
        """Load a trained model from its artifact."""

    def info(self) -> dict:
        return {"name": self.name, "display_name": self.display_name, "input_kind": self.input_kind,
                "is_baseline": self.is_baseline, "is_demo": self.is_demo}


def check_output(pred: np.ndarray, n: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape != (n, 2):
        raise ValueError(f"model returned shape {pred.shape}, expected ({n}, 2)")
    return pred
