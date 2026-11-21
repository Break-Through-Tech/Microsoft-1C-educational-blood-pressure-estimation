/**
 * Typed client for the CufflessAI API. Types mirror backend/schemas.py one-to-one; keep them in sync.
 * Units: mmHg for every blood-pressure value; predictions are ordered SBP then DBP.
 * Every response carries `mode` ("real" | "demo").
 */

export type Mode = "demo" | "real";
export type InputKind = "waveform" | "features";
export type SplitName = "validation" | "test";
export type Target = "sbp" | "dbp";

export const DISCLAIMER =
  "Educational prototype only. Not a medical device; do not use it to diagnose, monitor or treat any condition.";

// input limits for POST /api/predict (schemas.py: MIN_FS, MAX_FS, MIN_SECONDS, MAX_SECONDS)
export const MIN_FS = 12;
export const MAX_FS = 1000;
export const MIN_SECONDS = 5;
export const MAX_SECONDS = 120;
export const MAX_SIGNAL_SAMPLES = MAX_FS * MAX_SECONDS;

export interface ModeResponse {
  mode: Mode;
}

// GET /api/health
export interface HealthResponse extends ModeResponse {
  status: "ok";
  primary_model: string;
  torch_device: string;
  n_models: number;
  n_samples: number;
  notes: string[];
  disclaimer: string;
}

// GET /api/models
export interface TargetMetrics {
  n: number;
  mae: number;
  rmse: number;
}
export interface SplitMetrics {
  sbp: TargetMetrics;
  dbp: TargetMetrics;
}
export interface TargetBand {
  /** 90th percentile of |residual| */
  half_width_mmHg: number;
  /** share of test samples whose true value is inside the band */
  test_coverage: number | null;
}
export interface ModelBands {
  /** "validation", or "test (provisional)" while validation residuals are missing */
  band_source: string;
  quantile: number;
  sbp: TargetBand;
  dbp: TargetBand;
}
export interface ModelInfo {
  name: string;
  display_name: string;
  input_kind: InputKind;
  is_primary: boolean;
  is_baseline: boolean;
  is_demo: boolean;
  validation: SplitMetrics | null;
  test: SplitMetrics | null;
  bands: ModelBands | null;
}
export interface ModelsResponse extends ModeResponse {
  primary_model: string;
  selection_rule: string;
  models: ModelInfo[];
}

// GET /api/samples, /api/samples/{id}
export interface SampleSummary {
  id: number;
  subject_id: number;
  true_sbp: number;
  true_dbp: number;
}
export interface SamplesResponse extends ModeResponse {
  source: string;
  samples: SampleSummary[];
}
export interface SampleDetail extends ModeResponse {
  id: number;
  subject_id: number;
  true_sbp: number;
  true_dbp: number;
  fs: number;
  time_s: number[];
  /** cleaned PPG window, the exact model input (normalized to [-1, 1]) */
  amplitude: number[];
  /** unprocessed PPG of the same window, for display */
  raw_amplitude: number[] | null;
  source: string;
}

// POST /api/predict
/** Exactly one of: sample_id, or signal + fs. Submitted signals are never stored. */
export type PredictRequest =
  | { sample_id: number; signal?: never; fs?: never }
  | { sample_id?: never; signal: number[]; fs: number };

export interface Band {
  sbp_low: number;
  sbp_high: number;
  dbp_low: number;
  dbp_high: number;
  half_width_sbp: number;
  half_width_dbp: number;
  band_source: string;
  provisional: boolean;
}
export interface Prediction {
  model: string;
  display_name: string;
  is_baseline: boolean;
  sbp: number;
  dbp: number;
  band: Band | null;
}
export interface SignalQuality {
  ok: boolean;
  heart_rate_bpm: number | null;
  warnings: string[];
}
export interface InputSummary {
  source: "sample" | "signal";
  sample_id: number | null;
  n_samples: number;
  fs_in: number;
  duration_s: number;
  window_s: number;
}
export interface PredictResponse extends ModeResponse {
  input: InputSummary;
  /** reference label, only for dataset samples */
  true_sbp: number | null;
  true_dbp: number | null;
  quality: SignalQuality;
  primary: Prediction;
  all_models: Prediction[];
  disclaimer: string;
}

// POST /api/explain
export interface ExplainRequest {
  model: string;
  sample_id: number;
}
export interface SegmentImportance {
  start_s: number;
  end_s: number;
  sbp: number;
  dbp: number;
}
export interface FeatureImportance {
  name: string;
  sbp: number;
  dbp: number;
}
export interface ExplainResponse extends ModeResponse {
  model: string;
  sample_id: number;
  kind: "saliency" | "feature_importance" | "none";
  method: string;
  description: string | null;
  segments: SegmentImportance[] | null;
  features: FeatureImportance[] | null;
}

// GET /api/metrics
export interface MetricRow {
  model: string;
  display_name: string;
  is_baseline: boolean;
  split: SplitName;
  target: Target;
  n: number;
  mae: number;
  rmse: number;
  me: number | null;
  sd: number | null;
  band_half_width_p90: number | null;
  /** test rows only */
  band_coverage: number | null;
  band_source: string | null;
}
export interface MetricsResponse extends ModeResponse {
  primary_model: string;
  selection_rule: string;
  rows: MetricRow[];
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/** An error the UI can show as-is: `message` is already readable by a person. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI errors are {detail: string} or {detail: [{loc, msg, type}]}; turn either into one sentence. */
function readableDetail(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => String((d as { msg?: string }).msg ?? "").replace(/^Value error, /, ""))
      .filter(Boolean)
      .join("; ");
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch {
    throw new ApiError(0, "Can't reach the CufflessAI server. Check that the backend is running, then try again.");
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // non-JSON body (e.g. a proxy error page)
  }
  if (!res.ok) {
    const msg =
      readableDetail(body) ??
      (res.status >= 500
        ? `The server hit an error (${res.status}). Try again; if it keeps happening, check the backend logs.`
        : `Request failed (${res.status}).`);
    throw new ApiError(res.status, msg);
  }
  if (body === null) throw new ApiError(res.status, "The server sent an empty or unreadable response.");
  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<ModelsResponse>("/models"),
  samples: () => request<SamplesResponse>("/samples"),
  sample: (id: number) => request<SampleDetail>(`/samples/${id}`),
  metrics: () => request<MetricsResponse>("/metrics"),
  predict: (body: PredictRequest) =>
    request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify(body) }),
  explain: (body: ExplainRequest) =>
    request<ExplainResponse>("/explain", { method: "POST", body: JSON.stringify(body) }),
};
