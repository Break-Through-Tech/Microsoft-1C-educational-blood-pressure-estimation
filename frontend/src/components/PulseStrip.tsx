import { useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { Beats } from "@/api";
import { useChartColors } from "@/lib/tokens";

/**
 * Five seconds of PPG drawn like a clinical strip: light gridlines (1 s major, 0.2 s minor), no frame,
 * a circle on every detected beat and, on one representative beat, its onset, systolic peak and
 * dicrotic notch labelled with leader lines. Hover shows a crosshair with time and value.
 */
interface Props {
  time: number[];
  values: number[];
  fs: number;
  beats: Beats | null;
  /** label the representative beat (only meaningful on the cleaned signal the detector ran on) */
  landmarks: boolean;
  /** for screen readers */
  label: string;
  height?: number;
  /** changes when a different recording is shown, to replay the draw-in once */
  drawKey?: string | number;
}

// bottom margin holds a row for landmark labels, then the time axis
const M = { top: 34, right: 12, bottom: 46, left: 40 };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    setW(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function PulseStrip({ time, values, fs, beats, landmarks, label, height = 300, drawKey }: Props) {
  const c = useChartColors();
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  // the window's true length (625 samples at 125 Hz = 5 s), so the axis ends on a whole second
  const tMax = time.length ? time.length / fs : 5;
  const iw = Math.max(0, width - M.left - M.right);
  const ih = height - M.top - M.bottom;

  const [lo, hi] = useMemo(() => {
    let a = Infinity, b = -Infinity;
    for (const v of values) {
      if (v < a) a = v;
      if (v > b) b = v;
    }
    const pad = (b - a || 1) * 0.08;
    return [a - pad, b + pad];
  }, [values]);

  const x = (t: number) => M.left + (t / tMax) * iw;
  const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * ih;
  const at = (t: number) => values[Math.min(values.length - 1, Math.max(0, Math.round(t * fs)))];

  const path = useMemo(() => {
    if (!iw) return "";
    let d = "";
    for (let i = 0; i < values.length; i++) d += `${i ? "L" : "M"}${x(time[i]).toFixed(1)} ${y(values[i]).toFixed(1)}`;
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, time, iw, lo, hi]);

  const minor: number[] = [];
  for (let t = 0; t <= tMax + 1e-9; t += 0.2) minor.push(Math.round(t * 10) / 10);
  const major = minor.filter((t) => Math.abs(t - Math.round(t)) < 1e-6);
  const yTicks = niceTicks(lo, hi);

  const rep = landmarks ? beats?.representative ?? null : null;
  const repEnd = rep && beats ? beats.onsets_s.find((o) => o > rep.peak_s) ?? null : null;

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width) * tMax;
    setHover(Math.min(tMax, Math.max(0, t)));
  };

  return (
    <div ref={wrapRef} className="relative w-full select-none" role="img" aria-label={label}>
      {width > 0 && (
        <svg width={width} height={height} aria-hidden className="block overflow-visible">
          {/* grid: minor every 0.2 s, major every second, a few value lines */}
          {minor.map((t) => (
            <line key={`m${t}`} x1={x(t)} x2={x(t)} y1={M.top} y2={M.top + ih} stroke={c.rule} strokeOpacity={0.35} />
          ))}
          {major.map((t) => (
            <line key={`M${t}`} x1={x(t)} x2={x(t)} y1={M.top} y2={M.top + ih} stroke={c.rule} />
          ))}
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.left} x2={M.left + iw} y1={y(v)} y2={y(v)} stroke={c.rule} strokeOpacity={v === 0 ? 1 : 0.5} />
              <text x={M.left - 8} y={y(v)} dy="0.32em" textAnchor="end" className="num fill-muted-foreground text-[11px]">
                {fmtTick(v)}
              </text>
            </g>
          ))}
          {major.map((t) => (
            <text key={`xl${t}`} x={x(t)} y={M.top + ih + 38} textAnchor={t === 0 ? "start" : t >= tMax - 1e-6 ? "end" : "middle"} className="num fill-muted-foreground text-[11px]">
              {t} s
            </text>
          ))}

          {/* the representative beat's span */}
          {rep && repEnd != null && (
            <rect x={x(rep.onset_s)} y={M.top} width={x(repEnd) - x(rep.onset_s)} height={ih} fill={c.ink} fillOpacity={0.045} />
          )}

          <path
            key={drawKey}
            d={path}
            pathLength={1}
            fill="none"
            stroke={c.truth}
            strokeWidth={1.75}
            strokeLinejoin="round"
            strokeLinecap="round"
            style={{ strokeDasharray: 1, animation: "trace-draw 900ms cubic-bezier(.3,.6,.3,1) both" }}
          />

          {/* every detected beat */}
          {beats?.peaks_s.map((t) => (
            <circle key={`p${t}`} cx={x(t)} cy={y(at(t))} r={3.5} fill={c.surface} stroke={c.ink} strokeWidth={1.25} />
          ))}

          {rep && <Landmarks rep={rep} x={x} y={y} at={at} c={c} top={M.top} bottom={M.top + ih} left={M.left} right={M.left + iw} />}

          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + ih} stroke={c.muted} strokeWidth={1} />
              <circle cx={x(hover)} cy={y(at(hover))} r={3} fill={c.ink} />
            </g>
          )}
          <rect
            x={M.left}
            y={M.top}
            width={iw}
            height={ih}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
      )}
      {hover != null && width > 0 && (
        <div
          className="num pointer-events-none absolute top-0 rounded-[3px] bg-ink px-2 py-0.5 text-xs text-paper"
          style={{ left: Math.min(width - 96, Math.max(0, x(hover) - 40)) }}
        >
          {hover.toFixed(2)} s, {at(hover).toFixed(2)}
        </div>
      )}
      <div style={{ height: width > 0 ? 0 : height }} />
    </div>
  );
}

function Landmarks({
  rep,
  x,
  y,
  at,
  c,
  top,
  bottom,
  left,
  right,
}: {
  rep: { onset_s: number; peak_s: number; notch_s: number };
  x: (t: number) => number;
  y: (v: number) => number;
  at: (t: number) => number;
  c: ReturnType<typeof useChartColors>;
  top: number;
  bottom: number;
  left: number;
  right: number;
}) {
  const peak = { x: x(rep.peak_s), y: y(at(rep.peak_s)) };
  const notch = { x: x(rep.notch_s), y: y(at(rep.notch_s)) };
  const onset = { x: x(rep.onset_s), y: y(at(rep.onset_s)) };
  // Vertical leaders never cross the trace (one value per instant): up from the peak into the top
  // margin, down from the onset and notch into the label row under the plot. The onset label sits
  // left of its leader and the notch label right of its own, so the two never collide.
  const labelY = bottom + 15;
  const label = "fill-ink text-[12px] font-medium";
  const leader = { stroke: c.muted, strokeWidth: 1 };
  return (
    <g>
      <line x1={peak.x} y1={peak.y - 6} x2={peak.x} y2={top - 8} {...leader} />
      <text x={peak.x} y={top - 13} textAnchor="middle" className={label}>
        Systolic peak
      </text>

      <line x1={onset.x} y1={onset.y + 6} x2={onset.x} y2={labelY - 3} {...leader} />
      <text x={Math.max(left, onset.x + 4)} y={labelY} textAnchor="end" className={label} dx={-8}>
        Onset
      </text>

      <line x1={notch.x} y1={notch.y + 6} x2={notch.x} y2={labelY - 3} {...leader} />
      <text x={Math.min(right - 90, notch.x)} y={labelY} dx={4} className={label}>
        Dicrotic notch
      </text>

      {[peak, notch, onset].map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={4} fill={c.ink} stroke={c.surface} strokeWidth={2} />
      ))}
    </g>
  );
}

function niceTicks(lo: number, hi: number): number[] {
  const span = hi - lo;
  const raw = span / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

function fmtTick(v: number) {
  const s = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
  return s.replace("-", "−");
}
