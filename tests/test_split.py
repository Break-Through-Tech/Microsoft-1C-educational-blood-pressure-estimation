import json

import numpy as np
import pytest

from src.paths import Paths
from src.split import SPLIT_NAMES, assert_disjoint, load_splits, split_indices, split_subjects


def test_split_is_disjoint_and_complete_on_synthetic_ids():
    ids = np.repeat(np.arange(0, 3000, 3), 7)            # gaps and repeated windows, like real recordings
    splits = split_subjects(ids)
    assert_disjoint(splits)
    assert sorted(np.concatenate([splits[n] for n in SPLIT_NAMES]).tolist()) == np.unique(ids).tolist()
    idx = split_indices(ids, splits)
    assert sum(len(v) for v in idx.values()) == len(ids)


def test_split_matches_notebook_arithmetic():
    """Same permutation and cut points as model_training.ipynb, so seed 42 reproduces its split."""
    ids = np.arange(11_000)
    unique = np.random.default_rng(42).permutation(np.unique(ids))
    n_train, n_val = int(0.70 * len(unique)), int(0.15 * len(unique))
    splits = split_subjects(ids)
    assert set(splits["test"]) == set(unique[n_train + n_val:])
    assert set(splits["validation"]) == set(unique[n_train:n_train + n_val])


def test_assert_disjoint_catches_overlap():
    with pytest.raises(AssertionError):
        assert_disjoint({"train": [1, 2], "validation": [2], "test": [3]})


SPLITS = Paths().splits


@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.json not built")
def test_saved_split_has_no_subject_in_two_splits():
    splits = load_splits(SPLITS)                  # load_splits also asserts disjointness
    sets = [set(splits[n].tolist()) for n in SPLIT_NAMES]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.json not built")
def test_saved_split_reproduces_resnet_training_split():
    meta = json.loads(SPLITS.read_text())
    assert meta["verification"]["match"], meta["verification"]


@pytest.mark.skipif(not (SPLITS.exists() and Paths().test_samples.exists()), reason="evaluate.py not run")
def test_app_samples_come_from_test_subjects_only():
    splits = load_splits(SPLITS)
    with np.load(Paths().test_samples) as s:
        sample_ids = set(s["subject_ids"].tolist())
    assert sample_ids <= set(splits["test"].tolist())
    assert not sample_ids & set(splits["train"].tolist())
