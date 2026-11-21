import numpy as np

from src.paths import Paths
from src.registry import get_runtime


def test_demo_mode_when_artifacts_missing(tmp_path):
    rt = get_runtime(Paths(root=tmp_path), refresh=True)         # empty root: no data, artifacts or results
    assert rt.mode == "demo"
    assert rt.primary_model in rt.models
    X = rt.samples["X"]
    assert X.shape == (50, 1, 625) and X.dtype == np.float32
    for model in rt.models.values():
        assert model.is_demo
        pred = model.predict(X)
        assert pred.shape == (50, 2) and np.isfinite(pred).all()
        assert set(model.explain(X[:2])) >= {"method"}


def test_demo_bands_come_from_validation_residuals(tmp_path):
    rt = get_runtime(Paths(root=tmp_path), refresh=True)
    for name, band in rt.bands.items():
        assert band["band_source"] == "validation"
        assert band["sbp"]["half_width_mmHg"] > 0 and band["dbp"]["half_width_mmHg"] > 0


def test_demo_samples_are_seeded():
    from src.demo_data import make_demo_samples
    a, b = make_demo_samples(), make_demo_samples()
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["y"], b["y"])


def test_band_fallback_uses_test_residuals_provisionally():
    import pandas as pd
    from src.registry import compute_bands
    df = pd.DataFrame({"model": "m", "split": "test", "subject_id": range(10),
                       "true_sbp": np.arange(10.0), "pred_sbp": np.arange(10.0) + 2,
                       "true_dbp": np.zeros(10), "pred_dbp": np.ones(10)})
    bands = compute_bands(df)
    assert bands["m"]["band_source"] == "test (provisional)"
    assert np.isclose(bands["m"]["sbp"]["half_width_mmHg"], 2.0)
