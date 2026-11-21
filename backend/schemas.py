"""API contract. frontend/ types must mirror these models. Every response carries `mode`.

Units: mmHg for every blood-pressure value; predictions are ordered SBP then DBP.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Mode = Literal["demo", "real"]
InputKind = Literal["waveform", "features"]
SplitName = Literal["validation", "test"]
Target = Literal["sbp", "dbp"]

DISCLAIMER = ("Educational prototype only. Not a medical device; do not use it to diagnose, "
              "monitor or treat any condition.")

# input limits for POST /api/predict
# 12 Hz floor: the band-pass tops out at 5 Hz, so anything above 10 Hz keeps the pulse band; webcams
# often run at 15 fps in dim light. Everything is resampled to TARGET_FS before preprocessing.
MIN_FS, MAX_FS = 12.0, 1000.0          # Hz
MIN_SECONDS, MAX_SECONDS = 5.0, 120.0  # at least one full 5 s window, as in training
MAX_SIGNAL_SAMPLES = int(MAX_FS * MAX_SECONDS)


class ModeResponse(BaseModel):
    mode: Mode


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class HealthResponse(ModeResponse):
    status: Literal["ok"] = "ok"
    primary_model: str
    torch_device: str
    n_models: int
    n_samples: int
    notes: list[str] = []
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

class TargetMetrics(BaseModel):
    n: int
    mae: float
    rmse: float


class SplitMetrics(BaseModel):
    sbp: TargetMetrics
    dbp: TargetMetrics


class TargetBand(BaseModel):
    half_width_mmHg: float = Field(description="90th percentile of |residual|")
    test_coverage: Optional[float] = Field(None, description="share of test samples whose true value is inside the band")


class ModelBands(BaseModel):
    band_source: str = Field(description='"validation", or "test (provisional)" while validation residuals are missing')
    quantile: float
    sbp: TargetBand
    dbp: TargetBand


class ModelInfo(BaseModel):
    name: str
    display_name: str
    input_kind: InputKind
    is_primary: bool
    is_baseline: bool
    is_demo: bool
    validation: Optional[SplitMetrics] = None
    test: Optional[SplitMetrics] = None
    bands: Optional[ModelBands] = None


class ModelsResponse(ModeResponse):
    primary_model: str
    selection_rule: str
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# GET /api/samples, /api/samples/{id}
# ---------------------------------------------------------------------------

class SampleSummary(BaseModel):
    id: int
    subject_id: int
    true_sbp: float
    true_dbp: float


class SamplesResponse(ModeResponse):
    source: str
    samples: list[SampleSummary]


class SampleDetail(ModeResponse):
    id: int
    subject_id: int
    true_sbp: float
    true_dbp: float
    fs: float
    time_s: list[float]
    amplitude: list[float] = Field(description="cleaned PPG window, the exact model input (normalized to [-1, 1])")
    raw_amplitude: Optional[list[float]] = Field(None, description="unprocessed PPG of the same window, for display")
    source: str


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Exactly one of: sample_id, or signal + fs. Submitted signals are never stored."""
    sample_id: Optional[int] = Field(None, ge=0)
    signal: Optional[list[float]] = Field(None, max_length=MAX_SIGNAL_SAMPLES)
    fs: Optional[float] = Field(None, ge=MIN_FS, le=MAX_FS, description="sampling rate in Hz")

    @field_validator("signal")
    @classmethod
    def _finite(cls, v):
        if v is not None and not all(math.isfinite(x) for x in v):
            raise ValueError("signal must contain only finite numbers")
        return v

    @model_validator(mode="after")
    def _one_input(self):
        if (self.sample_id is None) == (self.signal is None):
            raise ValueError("provide exactly one of sample_id or signal")
        if self.signal is not None:
            if self.fs is None:
                raise ValueError("fs is required with signal")
            seconds = len(self.signal) / self.fs
            if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                raise ValueError(f"signal must be {MIN_SECONDS:g}-{MAX_SECONDS:g} s long, got {seconds:.2f} s")
            if max(self.signal) - min(self.signal) <= 1e-9:
                raise ValueError("signal is flat")
        return self


class Band(BaseModel):
    sbp_low: float
    sbp_high: float
    dbp_low: float
    dbp_high: float
    half_width_sbp: float
    half_width_dbp: float
    band_source: str
    provisional: bool


class Prediction(BaseModel):
    model: str
    display_name: str
    is_baseline: bool
    sbp: float
    dbp: float
    band: Optional[Band] = None


class SignalQuality(BaseModel):
    ok: bool
    heart_rate_bpm: Optional[float] = None
    warnings: list[str] = []


class InputSummary(BaseModel):
    source: Literal["sample", "signal"]
    sample_id: Optional[int] = None
    n_samples: int
    fs_in: float
    duration_s: float
    window_s: float


class PredictResponse(ModeResponse):
    input: InputSummary
    true_sbp: Optional[float] = Field(None, description="reference label, only for dataset samples")
    true_dbp: Optional[float] = None
    quality: SignalQuality
    primary: Prediction
    all_models: list[Prediction]
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# POST /api/explain
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    model: str
    sample_id: int = Field(ge=0)


class SegmentImportance(BaseModel):
    start_s: float
    end_s: float
    sbp: float
    dbp: float


class FeatureImportance(BaseModel):
    name: str
    sbp: float
    dbp: float


class ExplainResponse(ModeResponse):
    model: str
    sample_id: int
    kind: Literal["saliency", "feature_importance", "none"]
    method: str
    description: Optional[str] = None
    segments: Optional[list[SegmentImportance]] = None
    features: Optional[list[FeatureImportance]] = None


# ---------------------------------------------------------------------------
# GET /api/metrics
# ---------------------------------------------------------------------------

class MetricRow(BaseModel):
    model: str
    display_name: str
    is_baseline: bool
    split: SplitName
    target: Target
    n: int
    mae: float
    rmse: float
    me: Optional[float] = None
    sd: Optional[float] = None
    band_half_width_p90: Optional[float] = None
    band_coverage: Optional[float] = Field(None, description="test rows only")
    band_source: Optional[str] = None


class MetricsResponse(ModeResponse):
    primary_model: str
    selection_rule: str
    rows: list[MetricRow]
