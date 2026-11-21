import json

import numpy as np
import pytest

from src.demo_data import make_demo_samples
from src.paths import Paths
from src.preprocess import TARGET_FS, WINDOW_SIZE, clean_window

ARTIFACT = Paths().artifacts_dir / "resnet" / "resnet_bp_v3.pt"
NOTEBOOK = Paths().root / "notebooks" / "claire_resnet_pytorch" / "model_training.ipynb"
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(), reason="artifacts/resnet/resnet_bp_v3.pt not present")


@pytest.fixture(scope="module")
def model():
    from src.models.resnet import ResNetBP
    return ResNetBP.load(ARTIFACT)


@pytest.fixture(scope="module")
def windows():
    s = Paths().test_samples
    if s.exists():
        with np.load(s) as d:
            return d["X"][:64]
    return make_demo_samples(n=16)["X"]


@needs_artifact
def test_output_shape_and_plausible_mmhg(model, windows):
    pred = model.predict(windows)
    assert pred.shape == (len(windows), 2)
    assert np.isfinite(pred).all()
    assert (pred[:, 0] > 60).all() and (pred[:, 0] < 220).all(), "SBP outside plausible range"
    assert (pred[:, 1] > 30).all() and (pred[:, 1] < 140).all(), "DBP outside plausible range"
    assert (pred[:, 0] > pred[:, 1]).mean() > 0.95, "SBP should almost always exceed DBP"


@needs_artifact
def test_predict_is_deterministic_and_accepts_2d(model, windows):
    a = model.predict(windows[:4])
    b = model.predict(windows[:4, 0, :])
    np.testing.assert_allclose(a, b)


@needs_artifact
def test_rejects_wrong_window_length(model):
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 1, 500), dtype=np.float32))


@needs_artifact
def test_explain_gives_ten_segment_shares(model, windows):
    e = model.explain(windows[:3])
    assert e["method"] == "gradient_saliency"
    assert len(e["segments"]) == 10
    for t in ("sbp", "dbp"):
        shares = np.array(e[t])
        assert shares.shape == (3, 10)
        np.testing.assert_allclose(shares.sum(axis=1), 1, atol=1e-3)


@needs_artifact
@pytest.mark.skipif(not NOTEBOOK.exists(), reason="Claire's notebook not copied locally")
def test_matches_notebook_architecture(model, windows):
    """Our PPGResNet must give the same outputs as the class defined in Claire's notebook."""
    import torch
    import torch.nn as nn

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "class PPGResNet" in "".join(c["source"]))
    ns = {"torch": torch, "nn": nn, "WINDOW_SIZE": WINDOW_SIZE}
    exec(cell.split("\nmodel = build_model(CONFIG)")[0], ns)
    ckpt = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    ref = ns["build_model"](ckpt["config"])
    ref.load_state_dict(ckpt["model"])
    ref.eval()
    with torch.no_grad():
        expected = ref(torch.from_numpy(np.asarray(windows, dtype=np.float32))).numpy() * ckpt["y_std"] + ckpt["y_mean"]
    np.testing.assert_allclose(model.predict(windows), expected, rtol=1e-5, atol=1e-4)


def test_clean_window_resamples_and_fixes_length():
    t = np.arange(0, 8, 1 / 250)                                   # 8 s at 250 Hz
    x = 2 + np.sin(2 * np.pi * 1.2 * t)
    w = clean_window(x, fs=250)
    assert w.shape == (1, WINDOW_SIZE) and w.dtype == np.float32
    assert np.isclose(np.abs(w).max(), 1.0)
    short = clean_window(x[: int(3 * 250)], fs=250)                # 3 s -> zero-padded to 5 s
    assert short.shape == (1, WINDOW_SIZE)
    with pytest.raises(ValueError):
        clean_window(x[:100], fs=250)                              # 0.4 s is too short
    assert clean_window(np.sin(np.arange(0, 6, 1 / TARGET_FS) * 7), fs=TARGET_FS).shape == (1, WINDOW_SIZE)
