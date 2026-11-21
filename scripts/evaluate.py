"""Evaluate every registered model on the validation and test splits and pick the primary model.

    python scripts/evaluate.py [--n-samples 200]

For each ModelSpec in src/registry.py:
  * a model with fit() is trained on the train split and saved to its artifact path;
  * a pretrained model (no fit()) is loaded from its artifact and only evaluated if data/splits.json
    was verified to reproduce the split it was trained on (otherwise its test set could contain its
    own training recordings).
Primary model = lowest validation MAE_SBP + MAE_DBP among non-baseline models. Test metrics are
reported, never used for the choice.

Writes results/metrics.csv, results/residuals.csv (validation + test errors, column `split`),
results/best_model.json and data/test_samples.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import Paths                                                     # noqa: E402
from src.preprocess import features_matrix, load_dataset, read_raw_window       # noqa: E402
from src.registry import (MODEL_SPECS, compute_bands, error_metrics, predict_windows,  # noqa: E402
                          residuals_frame)
from src.split import load_splits, make_splits, split_indices                   # noqa: E402

# what the ResNet notebook reported for v3 (Claire-results/metrics.csv), for the comparison printout
REPORTED = {"resnet_v3": {"sbp": {"mae": 12.12, "rmse": 16.90}, "dbp": {"mae": 6.25, "rmse": 9.51}, "n": 79154}}


class WindowSubset:
    """Lazy view of X[idx] for fit(): len(), slicing and batches() without loading every window."""

    def __init__(self, X, idx):
        self.X, self.idx = X, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        return np.asarray(self.X[self.idx[item]])

    def batches(self, size: int = 8192):
        for s in range(0, len(self.idx), size):
            yield self[s:s + size]


def get_splits(paths: Paths, subject_ids: np.ndarray) -> dict:
    if paths.splits.exists():
        splits = load_splits(paths.splits)
        idx = split_indices(subject_ids, splits)
        if all(len(idx[k]) == splits["n_windows"][k] for k in idx):
            return splits
        print("data/splits.json does not match the current dataset cache; rebuilding it")
    return make_splits(paths)


def fit_or_load(spec, paths, X, y, idx, split_verified):
    model_cls, artifact = spec.cls, spec.artifact_path(paths)
    if hasattr(model_cls, "fit"):
        model = model_cls()
        train_X = (features_matrix(WindowSubset(X, idx["train"])[:]) if model.input_kind == "features"
                   else WindowSubset(X, idx["train"]))
        print(f"[{spec.name}] fitting on train split ({len(idx['train']):,} windows)")
        model.fit(train_X, y[idx["train"]])
        if hasattr(model, "save"):
            model.save(artifact)
        return model, "fitted on train split"
    if not artifact.exists():
        print(f"[{spec.name}] SKIPPED: artifact {artifact} not found")
        return None, None
    if not split_verified:
        print(f"[{spec.name}] SKIPPED: pretrained model, but data/splits.json is not verified to match its "
              "training split, so held-out recordings cannot be guaranteed")
        return None, None
    return model_cls.load(artifact), "pretrained; split verified to match its training split"


def pick_test_samples(rng, subject_ids, test_idx, n):
    """One window from each of n randomly chosen test recordings (so samples are not near-duplicates)."""
    recs = set(rng.choice(np.unique(subject_ids[test_idx]), size=n, replace=False).tolist())
    first = {}
    for i in rng.permutation(test_idx):
        s = int(subject_ids[i])
        if s in recs and s not in first:
            first[s] = i
    return np.sort(np.array(list(first.values())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=200, help="test windows saved for the app")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = Paths()
    t0 = time.time()
    X, subject_ids, y, meta = load_dataset(paths, return_meta=True)
    splits = get_splits(paths, subject_ids)
    idx = split_indices(subject_ids, splits)
    split_verified = bool(splits.get("verification", {}).get("match"))
    print(f"Dataset: {len(X):,} windows; " + ", ".join(f"{k} {len(v):,}" for k, v in idx.items())
          + f"; split verified: {split_verified}")

    metric_rows, residual_frames, provenance, predictions = [], [], {}, {}
    for spec in MODEL_SPECS:
        model, how = fit_or_load(spec, paths, X, y, idx, split_verified)
        if model is None:
            continue
        provenance[spec.name] = {"how": how, "artifact": str(spec.artifact_path(paths).relative_to(paths.root)),
                                 "is_baseline": model.is_baseline, "input_kind": model.input_kind}
        for split in ("validation", "test"):
            t1 = time.time()
            pred = predict_windows(model, X, idx[split])
            true = y[idx[split]]
            predictions[(spec.name, split)] = pred
            residual_frames.append(residuals_frame(spec.name, split, subject_ids[idx[split]], pred, true))
            for target, m in error_metrics(pred, true).items():
                metric_rows.append({"model": spec.name, "display_name": model.display_name, "split": split,
                                    "target": target, **m})
            print(f"[{spec.name}] {split}: MAE SBP {metric_rows[-2]['mae']:.2f}, DBP {metric_rows[-1]['mae']:.2f} "
                  f"mmHg ({time.time() - t1:.0f} s)")

    if not metric_rows:
        raise SystemExit("no model could be evaluated")
    metrics = pd.DataFrame(metric_rows)
    residuals = pd.concat(residual_frames, ignore_index=True)
    bands = compute_bands(residuals)
    metrics["band_half_width_p90"] = [bands[r.model][r.target]["half_width_mmHg"] for r in metrics.itertuples()]
    metrics["band_source"] = [bands[r.model]["band_source"] for r in metrics.itertuples()]
    metrics["band_coverage"] = [bands[r.model][r.target]["test_coverage"] if r.split == "test" else np.nan
                                for r in metrics.itertuples()]

    # primary model: validation only
    val = metrics[metrics.split == "validation"].pivot(index="model", columns="target", values="mae")
    val["combined"] = val["sbp"] + val["dbp"]
    candidates = val.loc[[m for m in val.index if not provenance[m]["is_baseline"]]].sort_values("combined")
    if candidates.empty:
        raise SystemExit("no non-baseline model was evaluated; cannot choose a primary model")
    primary = candidates.index[0]
    test = metrics[metrics.split == "test"].pivot(index="model", columns="target", values="mae")

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.metrics, index=False, float_format="%.4f")
    residuals.to_csv(paths.residuals, index=False, float_format="%.3f")
    best = {
        "primary_model": primary,
        "selection": "lowest validation MAE_SBP + MAE_DBP, baselines excluded; test metrics not used",
        "validation_mae": {m: {"sbp": round(r.sbp, 3), "dbp": round(r.dbp, 3), "combined": round(r.combined, 3)}
                           for m, r in val.iterrows()},
        "test_mae_report_only": {m: {"sbp": round(r.sbp, 3), "dbp": round(r.dbp, 3)} for m, r in test.iterrows()},
        "bands": bands,
        "models": provenance,
        "split": {"file": "data/splits.json", "verified": split_verified,
                  "n_windows": splits["n_windows"], "n_subjects": splits["n_subjects"]},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    paths.best_model.write_text(json.dumps(best, indent=1))

    # test samples for the app: one window per recording, with raw PPG / ABP for display
    rng = np.random.default_rng(args.seed)
    sel = pick_test_samples(rng, subject_ids, idx["test"], args.n_samples)
    raw = [read_raw_window(paths.raw_dir, int(meta["part"][i]), int(meta["record"][i]), int(meta["start"][i]))
           for i in sel]
    np.savez_compressed(paths.test_samples, X=np.asarray(X[sel]), y=y[sel], subject_ids=subject_ids[sel],
                        part=meta["part"][sel], record=meta["record"][sel], start=meta["start"][sel],
                        raw_ppg=np.stack([r["ppg"] for r in raw]), raw_abp=np.stack([r["abp"] for r in raw]),
                        source=np.array("UCI Cuff-Less BP Estimation, test split (held out from every model)"))

    print(f"\nPrimary model: {primary} (validation MAE SBP+DBP {candidates.combined.iloc[0]:.2f})")
    print(f"Wrote {paths.metrics}, {paths.residuals}, {paths.best_model}, {paths.test_samples} "
          f"({len(sel)} samples) in {time.time() - t0:.0f} s")

    # comparison with what the notebook reported
    for name, rep in REPORTED.items():
        if (name, "test") not in predictions:
            continue
        pred = predictions[(name, "test")]
        ours = error_metrics(pred, y[idx["test"]])
        print(f"\n{name} test, ours vs notebook-reported (n {len(pred):,} vs {rep['n']:,}):")
        for t in ("sbp", "dbp"):
            print(f"  {t.upper()}: MAE {ours[t]['mae']:.2f} vs {rep[t]['mae']:.2f}, "
                  f"RMSE {ours[t]['rmse']:.2f} vs {rep[t]['rmse']:.2f}")
        ref = paths.claire_reference_predictions
        if ref.exists():
            with np.load(ref) as r:
                if len(r["pred"]) == len(pred):
                    d = np.abs(pred - r["pred"])
                    print(f"  per-window difference from Claire's saved predictions: max {d.max():.4f}, "
                          f"mean {d.mean():.5f} mmHg")


if __name__ == "__main__":
    main()
