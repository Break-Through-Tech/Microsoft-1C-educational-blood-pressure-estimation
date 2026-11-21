import type { Prediction, PredictResponse } from "@/api";
import { BpRuler } from "@/components/BpRuler";
import { ChartFrame, Legend, type LegendKind } from "@/components/States";
import { mmHg } from "@/lib/format";
import { seriesColor, useChartColors } from "@/lib/tokens";
import { cn } from "@/lib/utils";

function offBy(p: Prediction, r: PredictResponse) {
  return Math.abs(p.sbp - r.true_sbp!) + Math.abs(p.dbp - r.true_dbp!);
}

/** Every model's estimate for the same input, on the same scales, in the fixed chart mapping. */
export function OtherModels({ result }: { result: PredictResponse }) {
  const c = useChartColors();
  const rows = result.all_models; // primary first, baselines last (backend order)
  const others = rows.filter((m) => m.model !== result.primary.model);
  const trainedOthers = others.filter((m) => !m.is_baseline);
  const hasRef = result.true_sbp != null && result.true_dbp != null;
  if (others.length === 0) return null;

  let otherIdx = 0;
  const styled = rows.map((m) => {
    const role = { is_primary: m.model === result.primary.model, is_baseline: m.is_baseline };
    const idx = role.is_primary || role.is_baseline ? 0 : otherIdx++;
    const variant = role.is_primary ? ("primary" as const) : role.is_baseline ? ("baseline" as const) : ("other" as const);
    return { m, variant, color: seriesColor(c, role, idx) };
  });

  const best = hasRef ? [...rows].sort((a, b) => offBy(a, result) - offBy(b, result))[0] : null;
  const title = best
    ? `${best.display_name} came closest to the measured value on this sample.`
    : "How every model reads the same signal.";

  return (
    <section className="mt-12">
      <ChartFrame
        title={title}
        subtitle="Each model's estimate and 90% range for the same 5-second window, on the same scales."
        notes={
          trainedOthers.length === 0
            ? `${result.primary.display_name} is the only trained model in the app so far. Teammates' models will appear here once they are retrained on the same split. The baseline predicts the training average for every recording, so it shows how much the model adds.`
            : "One sample says little on its own; the Compare page shows errors across thousands of recordings."
        }
      >
        <div className="overflow-x-auto px-2">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="text-left text-xs text-muted-foreground">
                <th className="w-[30%] pb-2 font-normal">Model</th>
                <th className="pb-2 font-normal">Systolic</th>
                <th className="pb-2 pl-6 font-normal">Diastolic</th>
              </tr>
            </thead>
            <tbody>
              {styled.map(({ m, variant, color }) => (
                <tr key={m.model} className="border-t border-rule">
                  <td className={cn("py-4 pr-4", variant === "baseline" ? "text-muted-foreground" : "font-semibold")}>{m.display_name}</td>
                  {(["sbp", "dbp"] as const).map((t) => (
                    <td key={t} className={t === "dbp" ? "py-4 pl-6" : "py-4"}>
                      <div className="flex items-center gap-4">
                        <span className="num w-9 shrink-0 text-right" style={{ color: variant === "baseline" ? undefined : color }}>
                          {mmHg(m[t])}
                        </span>
                        <BpRuler
                          size="mini"
                          target={t}
                          variant={variant}
                          color={color}
                          value={m[t]}
                          low={m.band?.[`${t}_low`]}
                          high={m.band?.[`${t}_high`]}
                          reference={hasRef ? (t === "sbp" ? result.true_sbp : result.true_dbp) : null}
                        />
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Legend
          className="mt-4 px-2"
          items={[
            ...styled.map(({ m, variant, color }) => ({
              kind: (variant === "primary" ? "primary-band" : variant === "baseline" ? "baseline" : "other") as LegendKind,
              label: m.display_name,
              color,
            })),
            ...(hasRef ? [{ kind: "truth-ring" as LegendKind, label: "Measured value" }] : []),
          ]}
        />
      </ChartFrame>
    </section>
  );
}
