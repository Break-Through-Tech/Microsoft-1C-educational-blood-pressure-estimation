/**
 * Camera pulse (remote PPG) helpers. Pure functions: no camera, no React.
 *
 * Each video frame is reduced to ONE number: the mean green value of a forehead patch. Blood absorbs
 * green light, so the patch gets slightly darker with each pulse. Frames arrive at uneven times, so the
 * series is resampled onto an even grid at the measured frame rate before it is sent.
 */

export interface Roi {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FaceGeometry {
  box: { originX: number; originY: number; width: number; height: number };
  /** normalized [0, 1] keypoints from the MediaPipe face detector: 0 = right eye, 1 = left eye */
  keypoints: { x: number; y: number }[];
}

/** A forehead patch above the eyebrows, from the eye positions (or the face box as a fallback). */
export function foreheadRoi(face: FaceGeometry, videoW: number, videoH: number): Roi | null {
  const [re, le] = face.keypoints;
  let roi: Roi;
  if (re && le) {
    const ex = ((re.x + le.x) / 2) * videoW;
    const ey = ((re.y + le.y) / 2) * videoH;
    const d = Math.hypot((re.x - le.x) * videoW, (re.y - le.y) * videoH);
    if (d < 8) return null;
    const w = 1.1 * d;
    const h = 0.45 * d;
    roi = { x: ex - w / 2, y: ey - 0.4 * d - h, w, h };
  } else {
    const { originX, originY, width, height } = face.box;
    roi = { x: originX + 0.3 * width, y: originY + 0.05 * height, w: 0.4 * width, h: 0.15 * height };
  }
  // clamp inside the frame
  const x = Math.max(0, Math.min(videoW - 2, roi.x));
  const y = Math.max(0, Math.min(videoH - 2, roi.y));
  const w = Math.max(2, Math.min(videoW - x, roi.w));
  const h = Math.max(2, Math.min(videoH - y, roi.h));
  return { x, y, w, h };
}

/** Smooth the patch between detections so it doesn't jitter (which would add noise to the signal). */
export function smoothRoi(prev: Roi | null, next: Roi, alpha = 0.3): Roi {
  if (!prev) return next;
  const m = (a: number, b: number) => a + alpha * (b - a);
  return { x: m(prev.x, next.x), y: m(prev.y, next.y), w: m(prev.w, next.w), h: m(prev.h, next.h) };
}

/** Mean of the green channel of RGBA pixel data. */
export function meanGreen(data: Uint8ClampedArray): number {
  let sum = 0;
  let n = 0;
  for (let i = 1; i < data.length; i += 4) {
    sum += data[i];
    n++;
  }
  return n ? sum / n : NaN;
}

export interface Sample {
  /** seconds */
  t: number;
  /** mean green, 0-255 */
  g: number;
}

/** Frames per second actually delivered, from the first and last timestamps. */
export function measuredFps(samples: Sample[]): number {
  if (samples.length < 2) return 0;
  const span = samples[samples.length - 1].t - samples[0].t;
  return span > 0 ? (samples.length - 1) / span : 0;
}

/**
 * Linear interpolation onto an even time grid at `fs`, and the sign flipped so peaks mean more blood
 * (the same orientation as the fingertip PPG the model was trained on).
 */
export function toEvenSignal(samples: Sample[], fs: number): number[] {
  if (samples.length < 2 || fs <= 0) return [];
  const t0 = samples[0].t;
  const t1 = samples[samples.length - 1].t;
  const n = Math.floor((t1 - t0) * fs) + 1;
  const out: number[] = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fs;
    while (j < samples.length - 2 && samples[j + 1].t < t) j++;
    const a = samples[j];
    const b = samples[j + 1];
    const f = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
    out[i] = -(a.g + Math.min(1, Math.max(0, f)) * (b.g - a.g));
  }
  return out;
}

/** Points for a live sparkline: the last `seconds` of the raw (un-flipped) series. */
export function recentWindow(samples: Sample[], seconds: number): Sample[] {
  if (samples.length === 0) return samples;
  const tEnd = samples[samples.length - 1].t;
  let i = samples.length - 1;
  while (i > 0 && samples[i - 1].t >= tEnd - seconds) i--;
  return samples.slice(i);
}
