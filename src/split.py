"""Train / validation / test split by subject (= UCI recording, see preprocess.py), saved to data/splits.json.

Reproduces the ResNet notebook exactly: the unique recording IDs that have at least one kept window are
permuted with np.random.default_rng(42); the first 70% train, the next 15% validation, the rest test.
The pretrained ResNet was trained on this train split, so evaluating it on validation / test only uses
recordings it never saw. `verify_against_reference` checks that against Claire's saved test predictions.

    python -m src.split          # build, verify and save data/splits.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import Paths

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
SPLIT_NAMES = ("train", "validation", "test")


def split_subjects(subject_ids: np.ndarray, seed: int = SEED, train: float = TRAIN_FRACTION,
                   val: float = VAL_FRACTION) -> dict[str, np.ndarray]:
    """Subject IDs per split. Same arithmetic as the notebook, so seed 42 gives its split."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(subject_ids))
    n_train = int(train * len(unique))
    n_val = int(val * len(unique))
    return {"train": np.sort(unique[:n_train]),
            "validation": np.sort(unique[n_train:n_train + n_val]),
            "test": np.sort(unique[n_train + n_val:])}


def split_indices(subject_ids: np.ndarray, splits: dict) -> dict[str, np.ndarray]:
    """Window indices per split, in dataset order (the order the notebook evaluated in)."""
    return {name: np.flatnonzero(np.isin(subject_ids, np.asarray(splits[name]))) for name in SPLIT_NAMES}


def assert_disjoint(splits: dict) -> None:
    sets = {name: set(np.asarray(splits[name]).tolist()) for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(f"{len(overlap)} subjects in both {a} and {b}, e.g. {sorted(overlap)[:5]}")


def verify_against_reference(y: np.ndarray, test_idx: np.ndarray, reference: Path) -> dict:
    """Compare the reproduced test windows with Claire's cached test labels (test_predictions.npz).

    The reference stores labels in her evaluation order, which is dataset order, so the reproduced
    split must give the same number of test windows and identical labels row by row.
    """
    if not reference.exists():
        return {"checked": False, "reason": f"{reference} not found"}
    with np.load(reference) as ref:
        ref_true = ref["true"]
    result = {"checked": True, "reference": str(reference), "reference_windows": int(len(ref_true)),
              "reproduced_windows": int(len(test_idx))}
    if len(ref_true) != len(test_idx):
        result.update(match=False, reason="different number of test windows")
        return result
    diff = np.abs(y[test_idx] - ref_true)
    result.update(max_label_diff_mmHg=float(diff.max()),
                  match=bool(diff.max() < 1e-3))
    if not result["match"]:
        result["reason"] = "test labels differ"
    return result


def save_splits(splits: dict, path: Path, meta: dict) -> None:
    assert_disjoint(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**meta, **{name: [int(s) for s in splits[name]] for name in SPLIT_NAMES}}
    path.write_text(json.dumps(out, indent=1))


def load_splits(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    for name in SPLIT_NAMES:
        d[name] = np.array(d[name], dtype=np.int64)
    assert_disjoint(d)
    return d


def make_splits(paths: Paths | None = None, allow_unverified: bool = False, verbose: bool = True) -> dict:
    """Build the split from the dataset, verify it reproduces the notebook's, and save it."""
    from .preprocess import CACHE_VERSION, load_dataset

    paths = paths or Paths()
    _, subject_ids, y = load_dataset(paths, verbose=verbose)
    splits = split_subjects(subject_ids)
    idx = split_indices(subject_ids, splits)
    check = verify_against_reference(y, idx["test"], paths.claire_reference_predictions)
    if verbose:
        print(f"Split verification: {check}")
    if not check.get("match") and not allow_unverified:
        raise RuntimeError("Reproduced split does not match the ResNet notebook's test set "
                           f"({check.get('reason')}). Not saving: the pretrained ResNet could be "
                           "evaluated on recordings it was trained on. See CLAUDE.md before overriding.")
    meta = {
        "unit": "subject = UCI recording id ((part-1)*3000 + record index); UCI has no patient IDs, "
                "so this is not guaranteed patient-independent",
        "method": f"np.random.default_rng({SEED}).permutation(unique ids with >=1 kept window); "
                  f"{TRAIN_FRACTION:.0%} / {VAL_FRACTION:.0%} / rest",
        "reproduces": "Claire's ResNet notebook split (model_training.ipynb / evaluation_plots.py)",
        "seed": SEED,
        "preprocessing": CACHE_VERSION,
        "n_windows": {name: int(len(idx[name])) for name in SPLIT_NAMES},
        "n_subjects": {name: int(len(splits[name])) for name in SPLIT_NAMES},
        "verification": check,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_splits(splits, paths.splits, meta)
    if verbose:
        print(f"Saved {paths.splits}: " + ", ".join(f"{n} {meta['n_subjects'][n]:,} recordings / "
                                                   f"{meta['n_windows'][n]:,} windows" for n in SPLIT_NAMES))
    return load_splits(paths.splits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="save even if the split does not match the notebook's (pretrained ResNet results "
                         "would then be invalid)")
    make_splits(allow_unverified=ap.parse_args().allow_unverified)


if __name__ == "__main__":
    main()
