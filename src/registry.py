"""Registered models and the runtime the app serves.

Adding a model = write a BPModel wrapper in src/models/ + add one ModelSpec line to MODEL_SPECS.
scripts/evaluate.py then fits it (if it has fit()), evaluates it and considers it for primary.

get_runtime() -> real mode when results/best_model.json, the primary model's artifact and
data/test_samples.npz exist; otherwise demo mode (placeholder models + synthetic windows).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models.base import BPModel
from .models.baseline import MeanBaseline
from .models.resnet import ResNetBP
from .paths import Paths
from .preprocess import features_matrix


@dataclass(frozen=True)
class ModelSpec:
    cls: type[BPModel]
    artifact: str                 # path under artifacts/

    @property
    def name(self) -> str:
        return self.cls.name

    def artifact_path(self, paths: Paths) -> Path:
        return paths.artifacts_dir / self.artifact


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(ResNetBP, "resnet/resnet_bp_v3.pt"),
    ModelSpec(MeanBaseline, "baseline/mean_baseline.json"),
]

BAND_QUANTILE = 0.90
TARGETS = ("sbp", "dbp")
SELECTION_RULE = "lowest validation MAE_SBP + MAE_DBP, baselines excluded; test metrics not used"
DEMO_SELECTION_RULE = "demo mode: placeholder model, nothing was selected"


# ---------------------------------------------------------------------------
# Helpers shared with scripts/evaluate.py
# ---------------------------------------------------------------------------

def prepare_inputs(model: BPModel, X: np.ndarray) -> np.ndarray:
    """Standard windows -> what this model's predict() expects."""
    return features_matrix(X) if model.input_kind == "features" else np.asarray(X, dtype=np.float32)


def predict_windows(model: BPModel, X: np.ndarray, idx: np.ndarray | None = None, chunk: int = 8192) -> np.ndarray:
    """Predict in chunks so a memory-mapped dataset is never loaded whole."""
    idx = np.arange(len(X)) if idx is None else idx
    out = [model.predict(prepare_inputs(model, X[idx[s:s + chunk]])) for s in range(0, len(idx), chunk)]
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)


def error_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    return {t: {"n": int(len(err)), "mae": float(np.abs(err[:, k]).mean()),
                "rmse": float(np.sqrt((err[:, k] ** 2).mean())), "me": float(err[:, k].mean()),
                "sd": float(err[:, k].std())} for k, t in enumerate(TARGETS)}


def compute_bands(residuals: pd.DataFrame, quantile: float = BAND_QUANTILE) -> dict:
    """Per model and target: band half-width = quantile of |residual| on the validation split
    (split-conformal style); test residuals only measure coverage.

    Temporary fallback (CLAUDE.md): a model without validation residuals gets its band from test
    residuals and band_source "test (provisional)".
    """
    bands = {}
    for model, df in residuals.groupby("model"):
        val, test = df[df.split == "validation"], df[df.split == "test"]
        source_df, source = (val, "validation") if len(val) else (test, "test (provisional)")
        entry = {"band_source": source, "quantile": quantile}
        for t in TARGETS:
            if not len(source_df):
                continue
            half = float(np.quantile(np.abs(source_df[f"pred_{t}"] - source_df[f"true_{t}"]), quantile))
            coverage = (float((np.abs(test[f"pred_{t}"] - test[f"true_{t}"]) <= half).mean())
                        if len(test) else None)
            entry[t] = {"half_width_mmHg": half, "test_coverage": coverage}
        bands[model] = entry
    return bands


def residuals_frame(model: str, split: str, subject_ids: np.ndarray, pred: np.ndarray, true: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"model": model, "split": split, "subject_id": subject_ids,
                         "true_sbp": true[:, 0], "pred_sbp": pred[:, 0],
                         "true_dbp": true[:, 1], "pred_dbp": pred[:, 1]})


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@dataclass
class Runtime:
    mode: str                                  # "real" | "demo"
    models: dict[str, BPModel]
    primary_model: str
    samples: dict                              # X, y, subject_ids, raw_ppg, ...
    metrics: pd.DataFrame
    residuals: pd.DataFrame
    bands: dict
    selection_rule: str = SELECTION_RULE
    notes: list[str] = field(default_factory=list)


def load_models(paths: Paths, specs: list[ModelSpec] = MODEL_SPECS) -> dict[str, BPModel]:
    """Every registered model whose artifact exists."""
    return {s.name: s.cls.load(s.artifact_path(paths)) for s in specs if s.artifact_path(paths).exists()}


def _real_runtime(paths: Paths) -> Runtime | None:
    if not (paths.best_model.exists() and paths.test_samples.exists() and paths.metrics.exists()):
        return None
    best = json.loads(paths.best_model.read_text())
    spec = next((s for s in MODEL_SPECS if s.name == best.get("primary_model")), None)
    if spec is None or not spec.artifact_path(paths).exists():
        return None
    models = load_models(paths)
    with np.load(paths.test_samples) as s:
        samples = {k: s[k] for k in s.files}
    residuals = pd.read_csv(paths.residuals) if paths.residuals.exists() else pd.DataFrame(
        columns=["model", "split", "subject_id", "true_sbp", "pred_sbp", "true_dbp", "pred_dbp"])
    return Runtime(mode="real", models=models, primary_model=spec.name, samples=samples,
                   metrics=pd.read_csv(paths.metrics), residuals=residuals,
                   bands=compute_bands(residuals), selection_rule=best.get("selection", SELECTION_RULE),
                   notes=[f"evaluated {best.get('created', '?')}; split verified: "
                          f"{best.get('split', {}).get('verified')}"])


def _demo_runtime() -> Runtime:
    from .demo_data import make_demo_samples
    from .models.demo import DemoPulseModel, demo_models

    samples = make_demo_samples()
    sid = samples["subject_ids"]
    # fake "validation" / "test" halves by subject so the band logic runs the same way as in real mode
    split = np.where(np.isin(sid, np.unique(sid)[: len(np.unique(sid)) // 2]), "validation", "test")
    models = demo_models(samples["y"][split == "validation"])
    frames, rows = [], []
    for name, m in models.items():
        pred = predict_windows(m, samples["X"])
        for sp in ("validation", "test"):
            mask = split == sp
            frames.append(residuals_frame(name, sp, sid[mask], pred[mask], samples["y"][mask]))
            for t, v in error_metrics(pred[mask], samples["y"][mask]).items():
                rows.append({"model": name, "split": sp, "target": t, **v})
    residuals = pd.concat(frames, ignore_index=True)
    return Runtime(mode="demo", models=models, primary_model=DemoPulseModel.name, samples=samples,
                   metrics=pd.DataFrame(rows), residuals=residuals, bands=compute_bands(residuals),
                   selection_rule=DEMO_SELECTION_RULE, notes=["Demo data: synthetic signals and placeholder models; numbers are not real results."])


_CACHE: dict[Path, Runtime] = {}


def get_runtime(paths: Paths | None = None, refresh: bool = False) -> Runtime:
    """Models, samples, metrics, residuals, bands, primary model and mode for the app."""
    paths = paths or Paths()
    if refresh or paths.root not in _CACHE:
        try:
            real = _real_runtime(paths)
        except Exception as e:  # noqa: BLE001 - a broken upload must not take the app down
            # e.g. a truncated .pt, malformed best_model.json or metrics.csv: serve demo mode and say why
            print(f"[cuffless] real mode unavailable, falling back to demo: {type(e).__name__}: {e}", file=sys.stderr)
            demo = _demo_runtime()
            demo.notes.append(f"Real results were found but could not be loaded ({type(e).__name__}); showing demo data.")
            _CACHE[paths.root] = demo
        else:
            _CACHE[paths.root] = real or _demo_runtime()
    return _CACHE[paths.root]
