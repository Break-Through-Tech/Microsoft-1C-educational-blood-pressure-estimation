"""The app's mean removal and band-pass must stay bit-identical to the notebook's neurokit2 calls."""
import numpy as np
import pytest

from src.preprocess import BANDPASS_HZ, FILTER_ORDER, TARGET_FS, bandpass_filter, remove_mean

nk = pytest.importorskip("neurokit2")


def _signals():
    rng = np.random.default_rng(0)
    t = np.arange(TARGET_FS * 30) / TARGET_FS
    return [rng.normal(0, 1, TARGET_FS * 120), np.sin(2 * np.pi * 1.2 * t) * 3 + 2 + 0.1 * t,
            rng.normal(5, 2, 40), np.full(700, 3.3), rng.normal(0, 1, 1250) * 1e3]


@pytest.mark.parametrize("i", range(5))
def test_remove_mean_matches_neurokit2(i):
    x = _signals()[i]
    assert np.array_equal(remove_mean(x), nk.signal_detrend(np.asarray(x, dtype=float), order=0))


@pytest.mark.parametrize("i", [0, 1, 3, 4])
def test_bandpass_matches_neurokit2(i):
    x = remove_mean(_signals()[i])
    expected = nk.signal_filter(x, sampling_rate=TARGET_FS, lowcut=BANDPASS_HZ[0], highcut=BANDPASS_HZ[1],
                                method="butterworth", order=FILTER_ORDER)
    assert np.array_equal(bandpass_filter(x), expected)
