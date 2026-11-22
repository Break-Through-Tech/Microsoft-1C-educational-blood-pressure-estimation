import type { Prediction, PredictResponse } from "@/api";
import { mmHg, signed } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Every model's estimate for the same input as one plain table: numbers only, primary first. */
export function OtherModels({ result }: { result: PredictResponse }) {
  const rows = result.all_models; // primary first, baselines last (backend order)
  const hasRef = result.true_sbp != null && result.true_dbp != null;
  if (rows.length < 2) return null;
  const trainedOthers = rows.filter((m) => m.model !== result.primary.model && !m.is_baseline);
  const off = (m: Prediction) => Math.abs(m.sbp - result.true_sbp!) + Math.abs(m.dbp - result.true_dbp!);
  const closest = hasRef ? rows.reduce((a, b) => (off(b) < off(a) ? b : a)).model : null;

  const th = "pb-2 text-xs font-normal text-muted-foreground";
  return (
    <section className="mt-12" aria-labelledby="all-models">
      <h2 id="all-models" className="mb-3 text-base font-semibold">
        All models
      </h2>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[560px] text-sm">
          <thead>
            <tr className="text-left">
              <th className={cn(th, "w-[34%]")}>Model</th>
              <th className={cn(th, "text-right")}>Estimate</th>
              {hasRef && <th className={cn(th, "text-right")}>Error</th>}
              <th className={cn(th, "text-right")}>Error band (90%)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const primary = m.model === result.primary.model;
              return (
                <tr key={m.model} className="border-t border-rule-soft">
                  <td className="py-2.5 pr-4">
                    <span className="flex items-center gap-2 whitespace-nowrap">
                      <span aria-hidden className={cn("h-3.5 w-[3px] rounded-[1px]", primary ? "bg-led" : "bg-transparent")} />
                      <span className={m.is_baseline ? "text-muted-foreground" : "text-ink"}>{m.display_name}</span>
                      {primary && <span className="text-xs text-muted-foreground">primary</span>}
                      {m.is_baseline && <span className="text-xs text-muted-foreground">training average</span>}
                    </span>
                  </td>
                  <td className={cn("py-2.5 text-right", primary ? "font-medium text-led" : m.is_baseline && "text-muted-foreground")}>
                    {mmHg(m.sbp)} / {mmHg(m.dbp)}
                  </td>
                  {hasRef && (
                    <td className={cn("py-2.5 text-right", m.model === closest ? "font-medium text-ink" : "text-muted-foreground")}>
                      {signed(m.sbp - result.true_sbp!)} / {signed(m.dbp - result.true_dbp!)}
                    </td>
                  )}
                  <td className="py-2.5 text-right text-muted-foreground">
                    {m.band ? `±${mmHg(m.band.half_width_sbp)} / ±${mmHg(m.band.half_width_dbp)}` : "–"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 max-w-[72ch] text-xs leading-relaxed text-muted-foreground">
        All values in mmHg, systolic / diastolic.{hasRef && " The darker error is the model closest on this recording."}{" "}
        {trainedOthers.length === 0
          ? `${result.primary.display_name} is the only trained model so far; teammates' models appear here once retrained on the same split.`
          : "One recording says little on its own; Compare shows errors across all held-out recordings."}
      </p>
    </section>
  );
}
