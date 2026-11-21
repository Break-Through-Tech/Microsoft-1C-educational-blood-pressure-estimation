"""Where every local-only file lives. Pass a different root (e.g. a temp dir in tests) to relocate all of them."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = Path(os.environ.get("CUFFLESS_ROOT", REPO_ROOT))

    # data/ (local only)
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        """UCI Cuff-Less BP Estimation: Part_1.mat ... Part_4.mat."""
        return self.data_dir / "cuff_less_bp_estimation"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def splits(self) -> Path:
        return self.data_dir / "splits.json"

    @property
    def test_samples(self) -> Path:
        return self.data_dir / "test_samples.npz"

    # artifacts/ and results/ (local only)
    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def metrics(self) -> Path:
        return self.results_dir / "metrics.csv"

    @property
    def residuals(self) -> Path:
        return self.results_dir / "residuals.csv"

    @property
    def best_model(self) -> Path:
        return self.results_dir / "best_model.json"

    # teammates' originals (local only, read-only)
    @property
    def claire_reference_predictions(self) -> Path:
        """Claire's cached v3 test predictions, used to verify the reproduced split."""
        return self.root / "notebooks" / "claire_resnet_pytorch" / "results" / "test_predictions.npz"
