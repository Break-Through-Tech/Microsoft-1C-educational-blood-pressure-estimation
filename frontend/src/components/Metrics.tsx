import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricRow, SplitName } from "@/api";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/misc";
import { err1, pct } from "@/lib/format";
import { BASELINE_DASH, MONO, seriesColor, useChartColors } from "@/lib/tokens";
import { Legend, type LegendKind } from "@/components/States";
import { cn } from "@/lib/utils";

export interface ModelMetrics {
  model: string;
  display_name: string;
  is_baseline: boolean;
  is_primary: boolean;
  sbp?: MetricRow;
  dbp?: MetricRow;
}

/** Pivot long rows into one entry per model for a split: primary first, baselines last. */
export function pivot(rows: MetricRow[], split: SplitName, primary: string): ModelMetrics[] {
  const by = new Map<string, ModelMetrics>();
  for (const r of rows) {
    if (r.split !== split) continue;
    const m = by.get(r.model) ?? {
      model: r.model,
      display_name: r.display_name,
      is_baseline: r.is_baseline,
      is_primary: r.model === primary,
    };
    m[r.target] = r;
    by.set(r.model, m);
  }
  return [...by.values()].sort(
    (a, b) => Number(b.is_primary) - Number(a.is_primary) || Number(a.is_baseline) - Number(b.is_baseline) || a.display_name.localeCompare(b.display_name),
  );
}

type SortKey = "sbp_mae" | "sbp_rmse" | "dbp_mae" | "dbp_rmse";
const COLS: { key: SortKey; label: string; target: "sbp" | "dbp"; stat: "mae" | "rmse" }[] = [
  { key: "sbp_mae", label: "Systolic MAE", target: "sbp", stat: "mae" },
  { key: "sbp_rmse", label: "Systolic RMSE", target: "sbp", stat: "rmse" },
  { key: "dbp_mae", label: "Diastolic MAE", target: "dbp", stat: "mae" },
  { key: "dbp_rmse", label: "Diastolic RMSE", target: "dbp", stat: "rmse" },
];

export function MetricsTable({ models, split }: { models: ModelMetrics[]; split: SplitName }) {
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return models;
    const col = COLS.find((c) => c.key === sort.key)!;
    const v = (m: ModelMetrics) => m[col.target]?.[col.stat] ?? Infinity;
    return [...models].sort((a, b) => (sort.asc ? v(a) - v(b) : v(b) - v(a)));
  }, [models, sort]);

  const toggle = (key: SortKey) =>
    setSort((s) => (s?.key === key ? (s.asc ? { key, asc: false } : null) : { key, asc: true }));

  return (
    <Table>
      <caption className="sr-only">
        Error in mmHg on the {split} split. Lower is better.
      </caption>
      <TableHeader>
        <TableRow>
          <TableHead className="pl-4">Model</TableHead>
          {COLS.map((c) => {
            const active = sort?.key === c.key;
            return (
              <TableHead key={c.key} className="text-right" aria-sort={active ? (sort!.asc ? "ascending" : "descending") : "none"}>
                <button type="button" onClick={() => toggle(c.key)} className="inline-flex items-center gap-1 rounded hover:text-ink">
                  {c.label}
                  {active && (sort!.asc ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
                </button>
              </TableHead>
            );
          })}
          <TableHead className="text-right">90% range</TableHead>
          {split === "test" && <TableHead className="text-right">Range coverage</TableHead>}
          <TableHead className="text-right">Windows</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((m) => (
          <TableRow key={m.model} className={cn(m.is_primary && "bg-led-soft/50")}>
            <TableCell className={cn("min-w-48 pl-4", m.is_primary && "border-l-[3px] border-l-led pl-[13px]")}>
              <div className="flex flex-wrap items-center gap-2">
                {m.is_baseline && <span aria-hidden className="inline-block h-3 w-5 rounded-[2px] border border-dashed border-baseline" />}
                <span className={m.is_baseline ? "text-muted-foreground" : "font-semibold"}>{m.display_name}</span>
                {m.is_primary && <Badge variant="led">Primary</Badge>}
              </div>
              {m.is_baseline && <p className="text-xs text-muted-foreground">Predicts the training average for everyone</p>}
            </TableCell>
            {COLS.map((c) => (
              <TableCell key={c.key} className={cn("num text-right", m.is_baseline && "text-muted-foreground")}>
                {m[c.target] ? err1(m[c.target]![c.stat]) : "n/a"}
              </TableCell>
            ))}
            <TableCell className="num text-right whitespace-nowrap text-muted-foreground">
              {m.sbp?.band_half_width_p90 != null && m.dbp?.band_half_width_p90 != null
                ? `±${Math.round(m.sbp.band_half_width_p90)} / ±${Math.round(m.dbp.band_half_width_p90)}`
                : "n/a"}
            </TableCell>
            {split === "test" && (
              <TableCell className="num text-right whitespace-nowrap text-muted-foreground">
                {m.sbp?.band_coverage != null && m.dbp?.band_coverage != null
                  ? `${pct(m.sbp.band_coverage)} / ${pct(m.dbp.band_coverage)}`
                  : "n/a"}
              </TableCell>
            )}
            <TableCell className="num text-right whitespace-nowrap text-muted-foreground">{(m.sbp?.n ?? m.dbp?.n)?.toLocaleString() ?? "n/a"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function MetricsBarChart({ models }: { models: ModelMetrics[] }) {
  const c = useChartColors();
  const data = COLS.map((col) => {
    const row: Record<string, number | string> = { metric: col.label };
    for (const m of models) {
      const r = m[col.target];
      if (r) row[m.model] = Math.round(r[col.stat] * 10) / 10;
    }
    return row;
  });
  let otherIdx = 0;
  const fills = models.map((m) => seriesColor(c, m, m.is_primary || m.is_baseline ? 0 : otherIdx++));

  return (
    <div>
      <div className="h-[320px] w-full" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 24, right: 8, bottom: 0, left: -8 }} barCategoryGap="22%" barGap={4}>
            <CartesianGrid stroke={c.rule} vertical={false} />
            <XAxis
              dataKey="metric"
              tickLine={false}
              axisLine={{ stroke: c.rule }}
              interval={0}
              height={44}
              tick={({ x, y, payload }: { x: number | string; y: number | string; payload: { value: string } }) => {
                const [target, stat] = String(payload.value).split(" ");
                return (
                  <text x={Number(x)} y={Number(y) + 14} textAnchor="middle" fill={c.muted} fontSize={12}>
                    <tspan x={Number(x)}>{target}</tspan>
                    <tspan x={Number(x)} dy={15} fill={c.ink}>
                      {stat}
                    </tspan>
                  </text>
                );
              }}
            />
            <YAxis
              tick={{ fill: c.muted, fontSize: 12, fontFamily: MONO }}
              tickLine={false}
              axisLine={false}
              width={56}
              label={{ value: "mmHg, lower is better", angle: -90, position: "insideLeft", offset: 18, fill: c.muted, fontSize: 12, style: { textAnchor: "middle" } }}
            />
            <Tooltip
              cursor={{ fill: c.rule, fillOpacity: 0.35 }}
              contentStyle={{ background: c.surface, border: `1px solid ${c.rule}`, borderRadius: 4, fontSize: 13 }}
              labelStyle={{ color: c.ink, fontWeight: 600 }}
              itemStyle={{ color: c.ink }}
              formatter={(v, name) => [`${Number(v).toFixed(1)} mmHg`, models.find((m) => m.model === name)?.display_name ?? String(name)]}
            />
            {models.map((m, i) => (
              <Bar
                key={m.model}
                dataKey={m.model}
                name={m.model}
                fill={m.is_baseline ? "transparent" : fills[i]}
                stroke={m.is_baseline ? c.baseline : "none"}
                strokeWidth={m.is_baseline ? 1.5 : 0}
                strokeDasharray={m.is_baseline ? BASELINE_DASH : undefined}
                radius={[2, 2, 0, 0]}
                maxBarSize={56}
                isAnimationActive={false}
              >
                <LabelList dataKey={m.model} position="top" fill={m.is_baseline ? c.muted : fills[i]} fontSize={12} fontFamily={MONO} formatter={(v) => Number(v).toFixed(1)} />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Legend
        className="mt-4 px-2"
        items={models.map((m, i) => ({
          kind: (m.is_primary ? "other" : m.is_baseline ? "baseline" : "other") as LegendKind,
          label: m.display_name,
          color: fills[i],
        }))}
      />
    </div>
  );
}
