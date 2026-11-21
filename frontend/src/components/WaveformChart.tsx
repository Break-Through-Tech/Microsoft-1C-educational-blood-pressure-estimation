import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SegmentImportance } from "@/api";
import { MONO, useChartColors } from "@/lib/tokens";

export interface WavePoint {
  t: number;
  v: number;
}

interface Props {
  points: WavePoint[];
  /** for screen readers: what this signal is */
  label: string;
  yLabel?: string;
  height?: number;
  /** saliency shading: darker = changed the estimate more */
  segments?: SegmentImportance[] | null;
  target?: "sbp" | "dbp";
}

export function toPoints(time: number[], values: number[]): WavePoint[] {
  return values.map((v, i) => ({ t: time[i], v }));
}

export function WaveformChart({ points, label, yLabel = "Amplitude", height = 220, segments, target = "sbp" }: Props) {
  const c = useChartColors();
  const tMax = points.length ? points[points.length - 1].t : 5;
  // shade relative to this sample's own spread, so the most and least influential regions differ visibly
  const [minShare, span] = useMemo(() => {
    const v = (segments ?? []).map((s) => s[target]);
    if (v.length === 0) return [0, 1];
    const lo = Math.min(...v);
    return [lo, Math.max(1e-9, Math.max(...v) - lo)];
  }, [segments, target]);
  const xTicks = useMemo(() => {
    const step = tMax > 30 ? 10 : tMax > 12 ? 2 : 1;
    const out: number[] = [];
    for (let t = 0; t <= tMax + 1e-9; t += step) out.push(Math.round(t * 100) / 100);
    return out;
  }, [tMax]);

  return (
    <div role="img" aria-label={label} className="w-full">
      <div style={{ height }} className="w-full" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
            <CartesianGrid stroke={c.rule} strokeDasharray="0" vertical={false} />
            {segments?.map((s, i) => (
              <ReferenceArea
                key={i}
                x1={s.start_s}
                x2={s.end_s}
                fill={c.primary}
                fillOpacity={0.03 + 0.5 * ((s[target] - minShare) / span)}
                stroke="none"
                ifOverflow="hidden"
              />
            ))}
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, tMax]}
              ticks={xTicks}
              tickFormatter={(t: number) => `${t} s`}
              stroke={c.muted}
              tick={{ fill: c.muted, fontSize: 12, fontFamily: MONO }}
              tickLine={false}
              axisLine={{ stroke: c.rule }}
            />
            <YAxis
              stroke={c.muted}
              tick={{ fill: c.muted, fontSize: 12, fontFamily: MONO }}
              tickLine={false}
              axisLine={false}
              width={52}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1))}
            />
            <Tooltip
              cursor={{ stroke: c.muted, strokeWidth: 1 }}
              contentStyle={{ background: c.surface, border: `1px solid ${c.rule}`, borderRadius: 4, fontSize: 13 }}
              labelStyle={{ color: c.muted }}
              itemStyle={{ color: c.ink }}
              labelFormatter={(t) => `${Number(t).toFixed(2)} s`}
              formatter={(v) => [Number(v).toFixed(3), yLabel]}
            />
            <Line type="monotone" dataKey="v" stroke={c.truth} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
