import { useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import type { SampleDetail } from "@/api";
import { Button } from "@/components/ui/button";

const W = 1000;
const H = 240;
const PAD_Y = 16;

/** Systolic peaks: local maxima above 0.4 at least 0.3 s apart (the window is scaled to [-1, 1]). */
export function findPeaks(amp: number[], fs: number): number[] {
  const minGap = Math.round(0.3 * fs);
  const peaks: number[] = [];
  for (let i = 1; i < amp.length - 1; i++) {
    if (amp[i] > 0.4 && amp[i] >= amp[i - 1] && amp[i] > amp[i + 1]) {
      const last = peaks[peaks.length - 1];
      if (last !== undefined && i - last < minGap) {
        if (amp[i] > amp[last]) peaks[peaks.length - 1] = i;
      } else peaks.push(i);
    }
  }
  return peaks;
}

export function beatStats(d: SampleDetail) {
  const peaks = findPeaks(d.amplitude, d.fs);
  const intervals = peaks.slice(1).map((p, i) => (p - peaks[i]) / d.fs);
  const mean = intervals.length ? intervals.reduce((a, b) => a + b, 0) / intervals.length : null;
  return { peaks, intervals, bpm: mean ? 60 / mean : null };
}

/**
 * Five seconds of a real PPG recording, drawn at the speed it was recorded, with one heartbeat
 * marked. The motion is the explanation: this is what the sensor sees, beat by beat.
 */
export function HeroTrace({ sample }: { sample: SampleDetail }) {
  const [run, setRun] = useState(0);
  const duration = sample.time_s[sample.time_s.length - 1] || 5;
  const { peaks, intervals } = useMemo(() => beatStats(sample), [sample]);

  const x = (t: number) => (t / duration) * W;
  const y = (v: number) => PAD_Y + ((1 - v) / 2) * (H - 2 * PAD_Y);
  const d = useMemo(
    () => "M" + sample.amplitude.map((v, i) => `${x(sample.time_s[i]).toFixed(1)},${y(v).toFixed(1)}`).join("L"),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sample],
  );

  // mark the second beat interval (the first is often cut by the window edge)
  const k = peaks.length >= 3 ? 1 : 0;
  const a = peaks[k];
  const b = peaks[k + 1];
  const tA = a !== undefined ? a / sample.fs : null;
  const tB = b !== undefined ? b / sample.fs : null;
  const interval = intervals[k] ?? null;
  const pct = (t: number) => `${(t / duration) * 100}%`;
  const ticks = Array.from({ length: Math.floor(duration) + 1 }, (_, i) => i);

  return (
    <div className="flex flex-col gap-2">
      <div className="relative" key={run}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[200px] w-full sm:h-[260px]" aria-hidden>
          {[-1, 0, 1].map((v) => (
            <line key={v} x1={0} x2={W} y1={y(v)} y2={y(v)} className="stroke-rule" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          ))}
          {tA !== null && tB !== null && (
            <rect
              x={x(tA)}
              width={x(tB) - x(tA)}
              y={0}
              height={H}
              className="fill-led-soft"
              style={{ animation: `fade-in .4s ease-out ${tB}s both` }}
            />
          )}
          <path
            d={d}
            fill="none"
            className="stroke-truth"
            strokeWidth={2}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
            pathLength={1}
            style={{ strokeDasharray: 1, animation: `trace-draw ${duration}s linear both` }}
          />
          {peaks.map((p) => (
            <circle
              key={p}
              cx={x(p / sample.fs)}
              cy={y(sample.amplitude[p])}
              r={4}
              className="fill-surface stroke-truth"
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
              style={{ animation: `fade-in .2s ease-out ${p / sample.fs}s both` }}
            />
          ))}
        </svg>
        {/* labels live in HTML so they don't stretch with the SVG */}
        {tA !== null && tB !== null && interval !== null && (
          <div
            className="pointer-events-none absolute top-1 text-xs"
            style={{ left: pct(tA), width: pct(tB - tA), animation: `fade-in .4s ease-out ${tB}s both` }}
          >
            <div className="mx-auto w-max rounded-[3px] bg-surface/90 px-1.5 py-0.5 text-center leading-tight">
              <span className="block font-semibold">One heartbeat</span>
              <span className="num text-muted-foreground">{interval.toFixed(2)} s</span>
            </div>
          </div>
        )}
      </div>
      <div className="num relative h-4 text-xs text-muted-foreground" aria-hidden>
        {ticks.map((t) => (
          <span key={t} className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full" style={{ left: pct(t) }}>
            {t} s
          </span>
        ))}
      </div>
      <div className="mt-2 flex justify-end">
        <Button variant="ghost" size="sm" onClick={() => setRun((r) => r + 1)}>
          <RotateCcw /> Replay at recording speed
        </Button>
      </div>
    </div>
  );
}
