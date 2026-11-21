"""FastAPI backend for the CufflessAI demo.

    .venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000

Loads src.registry.get_runtime() once at startup (real mode if evaluation results and artifacts
exist, otherwise demo mode). Submitted signals are processed in memory and never written or logged.

Production: after `npm run build` in frontend/, this server also serves frontend/dist (override with
CUFFLESS_FRONTEND_DIST) and answers unknown non-API paths with index.html for client-side routes.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.schemas import (Band, ExplainRequest, ExplainResponse, FeatureImportance,  # noqa: E402
                             HealthResponse, InputSummary, MetricRow, MetricsResponse, ModelBands,
                             ModelInfo, ModelsResponse, Prediction, PredictRequest, PredictResponse,
                             SampleDetail, SamplesResponse, SampleSummary, SegmentImportance,
                             SignalQuality, SplitMetrics, TargetBand, TargetMetrics)
from src.paths import Paths                                                          # noqa: E402
from src.preprocess import (TARGET_FS, WINDOW_SECONDS, WINDOW_SIZE, clean_window,    # noqa: E402
                            extract_features, good_ppg_window)
from src.registry import Runtime, get_runtime, prepare_inputs                        # noqa: E402

TORCH_DEVICE = "cpu"
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_metrics(rt: Runtime, model: str, split: str) -> SplitMetrics | None:
    df = rt.metrics[(rt.metrics.model == model) & (rt.metrics.split == split)]
    if set(df.target) != {"sbp", "dbp"}:
        return None
    row = {r.target: TargetMetrics(n=int(r.n), mae=float(r.mae), rmse=float(r.rmse)) for r in df.itertuples()}
    return SplitMetrics(**row)


def _model_bands(rt: Runtime, model: str) -> ModelBands | None:
    b = rt.bands.get(model)
    if not b or "sbp" not in b:
        return None
    return ModelBands(band_source=b["band_source"], quantile=b["quantile"],
                      sbp=TargetBand(**b["sbp"]), dbp=TargetBand(**b["dbp"]))


def _band(rt: Runtime, model: str, sbp: float, dbp: float) -> Band | None:
    b = rt.bands.get(model)
    if not b or "sbp" not in b:
        return None
    hs, hd = b["sbp"]["half_width_mmHg"], b["dbp"]["half_width_mmHg"]
    return Band(sbp_low=sbp - hs, sbp_high=sbp + hs, dbp_low=dbp - hd, dbp_high=dbp + hd,
                half_width_sbp=hs, half_width_dbp=hd, band_source=b["band_source"],
                provisional=b["band_source"] != "validation")


def _predict_all(rt: Runtime, X: np.ndarray) -> list[Prediction]:
    out = []
    for name, model in rt.models.items():
        sbp, dbp = (float(v) for v in model.predict(prepare_inputs(model, X))[0])
        out.append(Prediction(model=name, display_name=model.display_name, is_baseline=model.is_baseline,
                              sbp=sbp, dbp=dbp, band=_band(rt, name, sbp, dbp)))
    # primary first, baselines last
    return sorted(out, key=lambda p: (p.model != rt.primary_model, p.is_baseline, p.model))


def _quality(window: np.ndarray) -> SignalQuality:
    w = window.ravel()
    hr = extract_features(w)["heart_rate_bpm"]
    ok = bool(good_ppg_window(w))
    warnings = [] if ok else ["The pulse wave looks unusual (flat, noisy, or an implausible beat count); "
                              "the model was only trained on windows that pass this check."]
    return SignalQuality(ok=ok, heart_rate_bpm=None if np.isnan(hr) else round(hr, 1), warnings=warnings)


def _sample_index(rt: Runtime, sample_id: int) -> int:
    if not 0 <= sample_id < len(rt.samples["X"]):
        raise HTTPException(404, f"sample {sample_id} not found (0-{len(rt.samples['X']) - 1})")
    return sample_id


def _source(rt: Runtime) -> str:
    return str(rt.samples.get("source", "synthetic (demo)"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _mount_frontend(app: FastAPI, dist: Path) -> bool:
    """Serve the built React app. Registered last, so every /api route and /docs match first."""
    dist = dist.resolve()
    index = dist / "index.html"
    if not index.is_file():
        return False
    if (dist / "assets").is_dir():
        # Vite's content-hashed JS/CSS/fonts
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, f"unknown API route /{path}")
        if path:
            f = (dist / path).resolve()
            if f.is_file() and f.is_relative_to(dist):          # e.g. favicon.svg; never outside dist
                return FileResponse(f)
        # client-side route (/predict, /about, ...): the router in index.html takes over
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return True


def create_app(paths: Paths | None = None, frontend_dist: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = get_runtime(paths, refresh=True)     # load everything once
        yield

    app = FastAPI(title="CufflessAI API", version="0.1.0", lifespan=lifespan,
                  description="Educational blood-pressure estimation prototype. Not a medical device.")
    origins = [o.strip() for o in os.environ.get("CUFFLESS_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # report where and why, but never echo the submitted input back (it may be a user's signal,
        # and NaN / inf values in it cannot be encoded as JSON)
        detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def rt(request: Request) -> Runtime:
        return request.app.state.runtime

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request):
        r = rt(request)
        return HealthResponse(mode=r.mode, primary_model=r.primary_model, torch_device=TORCH_DEVICE,
                              n_models=len(r.models), n_samples=len(r.samples["X"]), notes=r.notes)

    @app.get("/api/models", response_model=ModelsResponse)
    def models(request: Request):
        r = rt(request)
        infos = [ModelInfo(name=name, display_name=m.display_name, input_kind=m.input_kind,
                           is_primary=name == r.primary_model, is_baseline=m.is_baseline, is_demo=m.is_demo,
                           validation=_split_metrics(r, name, "validation"), test=_split_metrics(r, name, "test"),
                           bands=_model_bands(r, name))
                 for name, m in r.models.items()]
        return ModelsResponse(mode=r.mode, primary_model=r.primary_model, selection_rule=r.selection_rule,
                              models=infos)

    @app.get("/api/samples", response_model=SamplesResponse)
    def samples(request: Request):
        r = rt(request)
        s = r.samples
        return SamplesResponse(mode=r.mode, source=_source(r), samples=[
            SampleSummary(id=i, subject_id=int(s["subject_ids"][i]), true_sbp=float(s["y"][i, 0]),
                          true_dbp=float(s["y"][i, 1])) for i in range(len(s["X"]))])

    @app.get("/api/samples/{sample_id}", response_model=SampleDetail)
    def sample(sample_id: int, request: Request):
        r = rt(request)
        i = _sample_index(r, sample_id)
        s = r.samples
        raw = s.get("raw_ppg")
        return SampleDetail(mode=r.mode, id=i, subject_id=int(s["subject_ids"][i]),
                            true_sbp=float(s["y"][i, 0]), true_dbp=float(s["y"][i, 1]), fs=TARGET_FS,
                            time_s=(np.arange(WINDOW_SIZE) / TARGET_FS).round(4).tolist(),
                            amplitude=np.asarray(s["X"][i]).ravel().round(5).tolist(),
                            raw_amplitude=None if raw is None else np.asarray(raw[i]).round(4).tolist(),
                            source=_source(r))

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        r = rt(request)
        true_sbp = true_dbp = None
        if body.sample_id is not None:
            i = _sample_index(r, body.sample_id)
            X = np.asarray(r.samples["X"][i:i + 1], dtype=np.float32)
            true_sbp, true_dbp = (float(v) for v in r.samples["y"][i])
            summary = InputSummary(source="sample", sample_id=i, n_samples=WINDOW_SIZE, fs_in=TARGET_FS,
                                   duration_s=WINDOW_SECONDS, window_s=WINDOW_SECONDS)
        else:
            # in memory only: the submitted signal is never written to disk or logged
            try:
                X = clean_window(np.asarray(body.signal, dtype=float), body.fs)[None]   # (1, 1, 625)
            except ValueError as e:
                raise HTTPException(422, str(e))
            summary = InputSummary(source="signal", n_samples=len(body.signal), fs_in=body.fs,
                                   duration_s=round(len(body.signal) / body.fs, 3), window_s=WINDOW_SECONDS)
        preds = _predict_all(r, X)
        return PredictResponse(mode=r.mode, input=summary, true_sbp=true_sbp, true_dbp=true_dbp,
                               quality=_quality(X[0]), primary=preds[0], all_models=preds)

    @app.post("/api/explain", response_model=ExplainResponse)
    def explain(body: ExplainRequest, request: Request):
        r = rt(request)
        if body.model not in r.models:
            raise HTTPException(404, f"unknown model {body.model!r}; choose from {sorted(r.models)}")
        i = _sample_index(r, body.sample_id)
        model = r.models[body.model]
        X = np.asarray(r.samples["X"][i:i + 1], dtype=np.float32)
        e = model.explain(prepare_inputs(model, X))         # the only place gradients are enabled
        common = dict(mode=r.mode, model=body.model, sample_id=i, method=e.get("method", "none"),
                      description=e.get("description"))
        if "features" in e:
            feats = [FeatureImportance(name=k, sbp=float(v["sbp"]), dbp=float(v["dbp"])) for k, v in e["features"].items()]
            return ExplainResponse(kind="feature_importance", features=feats, **common)
        if "segments" in e:
            segs = [SegmentImportance(start_s=s["start_s"], end_s=s["end_s"], sbp=e["sbp"][0][k], dbp=e["dbp"][0][k])
                    for k, s in enumerate(e["segments"])]
            return ExplainResponse(kind="saliency", segments=segs, **common)
        return ExplainResponse(kind="none", **common)

    @app.get("/api/metrics", response_model=MetricsResponse)
    def metrics(request: Request):
        r = rt(request)
        rows = []
        for m in r.metrics.to_dict("records"):
            model = r.models.get(m["model"])
            band = r.bands.get(m["model"], {})
            tb = band.get(m["target"], {})
            rows.append(MetricRow(
                model=m["model"], display_name=model.display_name if model else m.get("display_name", m["model"]),
                is_baseline=bool(model.is_baseline) if model else False, split=m["split"], target=m["target"],
                n=int(m["n"]), mae=float(m["mae"]), rmse=float(m["rmse"]),
                me=_num(m.get("me")), sd=_num(m.get("sd")),
                band_half_width_p90=tb.get("half_width_mmHg"),
                band_coverage=tb.get("test_coverage") if m["split"] == "test" else None,
                band_source=band.get("band_source")))
        return MetricsResponse(mode=r.mode, primary_model=r.primary_model, selection_rule=r.selection_rule, rows=rows)

    if frontend_dist is None:
        frontend_dist = Path(os.environ.get("CUFFLESS_FRONTEND_DIST", DEFAULT_FRONTEND_DIST))
    app.state.serves_frontend = _mount_frontend(app, frontend_dist)
    return app


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


app = create_app()
