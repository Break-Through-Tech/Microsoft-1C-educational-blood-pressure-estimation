"""Every endpoint, in demo mode (empty temp root) and real mode (repo root, skipped without results)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.schemas import DISCLAIMER
from src.paths import Paths

REAL = Paths()
REAL_READY = REAL.best_model.exists() and REAL.test_samples.exists() and (REAL.artifacts_dir / "resnet").exists()


@pytest.fixture(scope="module", params=["demo", "real"])
def client(request, tmp_path_factory):
    if request.param == "real" and not REAL_READY:
        pytest.skip("real mode needs results/, artifacts/ and data/test_samples.npz (run scripts/evaluate.py)")
    root = REAL.root if request.param == "real" else tmp_path_factory.mktemp("empty_root")
    with TestClient(create_app(Paths(root=root))) as c:
        c.expected_mode = request.param
        c.root = root
        yield c


def _ppg(seconds=10.0, fs=250.0, hr=72.0):
    t = np.arange(int(seconds * fs)) / fs
    beat = 60 / hr
    phase = (t % beat) / beat
    return (2 + np.exp(-0.5 * ((phase - 0.15) / 0.07) ** 2) + 0.35 * np.exp(-0.5 * ((phase - 0.45) / 0.09) ** 2)
            + 0.2 * np.sin(2 * np.pi * 0.2 * t)).tolist()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == client.expected_mode and d["status"] == "ok" and d["torch_device"] == "cpu"
    assert d["disclaimer"] == DISCLAIMER
    if client.expected_mode == "real":
        assert d["primary_model"] == "resnet_v3"


def test_models(client):
    d = client.get("/api/models").json()
    assert d["mode"] == client.expected_mode and d["selection_rule"]
    primaries = [m for m in d["models"] if m["is_primary"]]
    assert len(primaries) == 1 and primaries[0]["name"] == d["primary_model"]
    assert any(m["is_baseline"] for m in d["models"])
    for m in d["models"]:
        for split in ("validation", "test"):
            for t in ("sbp", "dbp"):
                assert m[split][t]["mae"] > 0 and m[split][t]["rmse"] >= m[split][t]["mae"]
        assert m["bands"]["sbp"]["half_width_mmHg"] > 0
    if client.expected_mode == "real":
        resnet = next(m for m in d["models"] if m["name"] == "resnet_v3")
        assert resnet["test"]["sbp"]["mae"] == pytest.approx(12.12, abs=0.01)
        assert resnet["test"]["dbp"]["mae"] == pytest.approx(6.25, abs=0.01)
        assert resnet["bands"]["band_source"] == "validation"


def test_samples_list_and_detail(client):
    d = client.get("/api/samples").json()
    assert d["mode"] == client.expected_mode
    n = 200 if client.expected_mode == "real" else 50
    assert len(d["samples"]) == n
    s = d["samples"][3]
    detail = client.get(f"/api/samples/{s['id']}").json()
    assert detail["mode"] == client.expected_mode
    assert len(detail["time_s"]) == len(detail["amplitude"]) == 625
    assert detail["fs"] == 125 and max(abs(v) for v in detail["amplitude"]) <= 1.0 + 1e-6
    assert detail["true_sbp"] == pytest.approx(s["true_sbp"]) and detail["subject_id"] == s["subject_id"]
    assert client.get(f"/api/samples/{n}").status_code == 404


def test_predict_sample(client):
    d = client.post("/api/predict", json={"sample_id": 0}).json()
    assert d["mode"] == client.expected_mode and d["disclaimer"] == DISCLAIMER
    assert d["input"]["source"] == "sample" and d["true_sbp"] is not None
    p = d["primary"]
    assert p["model"] == d["all_models"][0]["model"] and not p["is_baseline"]
    assert 40 < p["sbp"] < 250 and 20 < p["dbp"] < 160
    assert p["band"]["sbp_low"] < p["sbp"] < p["band"]["sbp_high"]
    assert p["band"]["provisional"] is False
    assert any(m["is_baseline"] for m in d["all_models"])


def test_predict_signal_is_cleaned_and_not_stored(client):
    before = {p for p in client.root.rglob("*")} if client.expected_mode == "demo" else None
    d = client.post("/api/predict", json={"signal": _ppg(fs=250), "fs": 250}).json()
    assert d["mode"] == client.expected_mode
    assert d["input"]["source"] == "signal" and d["input"]["fs_in"] == 250 and d["true_sbp"] is None
    assert d["quality"]["ok"] and 65 < d["quality"]["heart_rate_bpm"] < 80
    assert len(d["all_models"]) >= 2
    if before is not None:                        # nothing written anywhere under the app's root
        assert {p for p in client.root.rglob("*")} == before


@pytest.mark.parametrize("body", [
    {},                                                        # neither input
    {"sample_id": 0, "signal": [1.0] * 1000, "fs": 125},       # both inputs
    {"signal": [1.0, 2.0] * 400},                              # missing fs
    {"signal": _ppg(fs=125)[:400], "fs": 125},                 # 3.2 s: too short
    {"signal": _ppg(seconds=130, fs=125), "fs": 125},          # 130 s: too long
    {"signal": _ppg(fs=125), "fs": 5},                         # fs out of range
    {"signal": _ppg(fs=125), "fs": 5000},
    {"signal": [0.5] * 1250, "fs": 125},                       # flat
    {"sample_id": -1},
])
def test_predict_rejects_invalid_input(client, body):
    assert client.post("/api/predict", json=body).status_code == 422


def test_predict_accepts_webcam_frame_rate(client):
    """A 15 fps camera signal (dim-light webcam) is resampled to TARGET_FS and scored like any other."""
    r = client.post("/api/predict", json={"signal": _ppg(seconds=15, fs=15.0), "fs": 15.0})
    assert r.status_code == 200, r.text
    assert r.json()["input"]["fs_in"] == 15.0


def test_predict_rejects_non_finite_values(client):
    sig = _ppg(fs=125)
    raw = "[" + ",".join(["NaN"] + [repr(v) for v in sig[1:]]) + "]"
    r = client.post("/api/predict", content='{"fs": 125, "signal": ' + raw + "}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert all("input" not in e for e in r.json()["detail"]), "submitted signal must not be echoed back"


def test_predict_unknown_sample(client):
    assert client.post("/api/predict", json={"sample_id": 10_000}).status_code == 404


def test_explain(client):
    primary = client.get("/api/health").json()["primary_model"]
    d = client.post("/api/explain", json={"model": primary, "sample_id": 1}).json()
    assert d["mode"] == client.expected_mode and d["kind"] == "saliency"
    assert len(d["segments"]) == 10
    assert sum(s["sbp"] for s in d["segments"]) == pytest.approx(1, abs=1e-2)
    if client.expected_mode == "real":
        assert d["method"] == "gradient_saliency"
    baseline = client.post("/api/explain", json={"model": "mean_baseline", "sample_id": 1}).json()
    assert baseline["kind"] == "none"
    assert client.post("/api/explain", json={"model": "nope", "sample_id": 1}).status_code == 404
    assert client.post("/api/explain", json={"model": primary, "sample_id": 99_999}).status_code == 404


def test_metrics(client):
    d = client.get("/api/metrics").json()
    assert d["mode"] == client.expected_mode
    rows = d["rows"]
    assert {(r["split"], r["target"]) for r in rows} == {(s, t) for s in ("validation", "test") for t in ("sbp", "dbp")}
    assert any(r["is_baseline"] for r in rows)
    for r in rows:
        assert r["band_half_width_p90"] > 0
        assert (r["band_coverage"] is not None) == (r["split"] == "test")


def test_cors_allows_vite_dev_server(client):
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.options("/api/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
