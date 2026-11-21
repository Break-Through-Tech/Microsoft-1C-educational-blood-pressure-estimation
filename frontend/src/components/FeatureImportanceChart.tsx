import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FeatureImportance } from "@/api";
import { MONO, useChartColors } from "@/lib/tokens";

/** Horizontal bars, sorted, for feature-based models. Shows the top 15 for the chosen target. */
export function FeatureImportanceChart({ features, target, color }: { features: FeatureImportance[]; target: "sbp" | "dbp"; color?: string }) {
  const c = useChartColors();
  const data = [...features].sort((a, b) => b[target] - a[target]).slice(0, 15);
  const height = Math.max(160, data.length * 30 + 40);
  return (
    <div role="img" aria-label={`Feature importance for ${target === "sbp" ? "systolic" : "diastolic"} pressure`}>
      <div style={{ height }} aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }} barCategoryGap={6}>
            <CartesianGrid stroke={c.rule} horizontal={false} />
            <XAxis type="number" tick={{ fill: c.muted, fontSize: 12, fontFamily: MONO }} tickLine={false} axisLine={{ stroke: c.rule }} />
            <YAxis type="category" dataKey="name" width={170} tick={{ fill: c.ink, fontSize: 12 }} tickLine={false} axisLine={false} />
            <Tooltip
              cursor={{ fill: c.rule, fillOpacity: 0.35 }}
              contentStyle={{ background: c.surface, border: `1px solid ${c.rule}`, borderRadius: 4, fontSize: 13 }}
              itemStyle={{ color: c.ink }}
              formatter={(v) => [Number(v).toFixed(3), "Importance"]}
            />
            <Bar dataKey={target} fill={color ?? c.primary} radius={[0, 2, 2, 0]} isAnimationActive={false}>
              <LabelList dataKey={target} position="right" fill={c.muted} fontSize={12} fontFamily={MONO} formatter={(v) => Number(v).toFixed(2)} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {features.length > data.length && (
        <p className="mt-2 text-xs text-muted-foreground">
          Showing the {data.length} most important of {features.length} features.
        </p>
      )}
    </div>
  );
}
