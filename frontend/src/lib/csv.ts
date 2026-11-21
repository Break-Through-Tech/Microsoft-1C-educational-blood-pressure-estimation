/**
 * Minimal in-browser CSV reader for a PPG recording: one or more numeric columns, optional header.
 * The file never leaves the browser except as the numeric signal sent once to /api/predict.
 */
import { MAX_FS, MAX_SECONDS, MIN_FS, MIN_SECONDS } from "@/api";

export interface CsvColumn {
  name: string;
  values: number[];
  /** rows whose cell was empty or not a number */
  invalid: number;
}

export interface ParsedCsv {
  columns: CsvColumn[];
  rows: number;
  /** sampling rate implied by a time column, if one was found */
  inferredFs: number | null;
  timeColumn: string | null;
}

const TIME_NAMES = /^(t|time|times|time_s|seconds|sec|s|timestamp)$/i;

function detectDelimiter(line: string): string {
  const counts = [",", ";", "\t", " "].map((d) => [d, line.split(d).length - 1] as const);
  const best = counts.reduce((a, b) => (b[1] > a[1] ? b : a));
  return best[1] > 0 ? best[0] : ",";
}

function splitLine(line: string, delim: string): string[] {
  const cells = delim === " " ? line.trim().split(/\s+/) : line.split(delim);
  return cells.map((c) => c.trim().replace(/^"(.*)"$/, "$1"));
}

const isNum = (s: string) => s !== "" && Number.isFinite(Number(s));

export function parseCsv(text: string): ParsedCsv {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "" && !l.trimStart().startsWith("#"));
  if (lines.length === 0) throw new Error("The file is empty.");
  const delim = detectDelimiter(lines[0]);
  const first = splitLine(lines[0], delim);
  const hasHeader = first.some((c) => !isNum(c));
  const width = first.length;
  const names = hasHeader ? first.map((n, i) => n || `Column ${i + 1}`) : first.map((_, i) => `Column ${i + 1}`);
  const body = hasHeader ? lines.slice(1) : lines;
  if (body.length === 0) throw new Error("The file has a header row but no data rows.");

  const cols: CsvColumn[] = names.map((name) => ({ name, values: [], invalid: 0 }));
  for (const line of body) {
    const cells = splitLine(line, delim);
    for (let c = 0; c < width; c++) {
      const s = cells[c] ?? "";
      if (isNum(s)) cols[c].values.push(Number(s));
      else cols[c].invalid++;
    }
  }
  // keep columns that are mostly numeric
  const numeric = cols.filter((c) => c.values.length >= body.length * 0.95 && c.values.length > 1);
  if (numeric.length === 0) throw new Error("No column of numbers was found. The file needs at least one numeric column.");

  const time = numeric.find((c) => TIME_NAMES.test(c.name)) ?? null;
  let inferredFs: number | null = null;
  if (time && time.values.length > 2) {
    const diffs = time.values.slice(1).map((v, i) => v - time.values[i]).sort((a, b) => a - b);
    const median = diffs[Math.floor(diffs.length / 2)];
    if (median > 0) {
      // times in milliseconds are common; prefer the reading that gives a plausible rate
      const asSeconds = 1 / median;
      const asMs = 1000 / median;
      inferredFs = asSeconds >= MIN_FS && asSeconds <= MAX_FS ? asSeconds : asMs >= MIN_FS && asMs <= MAX_FS ? asMs : null;
      if (inferredFs) inferredFs = Math.round(inferredFs * 100) / 100;
    }
  }
  const signals = numeric.filter((c) => c !== time);
  if (signals.length === 0) throw new Error("Only a time column was found. Add a column with the PPG signal.");
  return { columns: signals, rows: body.length, inferredFs, timeColumn: time?.name ?? null };
}

/** Same checks as PredictRequest in backend/schemas.py, worded as a fix. Returns null when it's fine. */
export function signalProblem(values: number[], fs: number): string | null {
  if (!Number.isFinite(fs) || fs < MIN_FS || fs > MAX_FS)
    return `Set a sampling rate between ${MIN_FS} and ${MAX_FS} Hz.`;
  const seconds = values.length / fs;
  if (seconds < MIN_SECONDS)
    return `This signal is ${seconds.toFixed(1)} s at ${fs} Hz. Add at least ${(MIN_SECONDS - seconds).toFixed(1)} s more; the model needs one full ${MIN_SECONDS} s window.`;
  if (seconds > MAX_SECONDS)
    return `This signal is ${seconds.toFixed(0)} s long. Trim it to ${MAX_SECONDS} s or less.`;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo <= 1e-9) return "This column is flat (every value is the same). Pick the column with the pulse signal.";
  return null;
}

/** Thin a long signal for plotting only; the full signal is what gets sent. */
export function decimateForPlot(values: number[], fs: number, maxPoints = 1500) {
  const step = Math.max(1, Math.ceil(values.length / maxPoints));
  const out: { t: number; v: number }[] = [];
  for (let i = 0; i < values.length; i += step) out.push({ t: i / fs, v: values[i] });
  return out;
}
