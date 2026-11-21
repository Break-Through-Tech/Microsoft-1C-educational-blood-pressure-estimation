import { TriangleAlert } from "lucide-react";
import type { Prediction, PredictResponse } from "@/api";
import { BpRuler } from "@/components/BpRuler";
import { ChartFrame, Legend } from "@/components/States";
import { Alert, Skeleton } from "@/components/ui/misc";
import { mmHg } from "@/lib/format";

function Reading({ label, target, p, reference }: { label: string; target: "sbp" | "dbp"; p: Prediction; reference: number | null }) {
  const value = p[target];
  const lo = p.band ? p.band[`${target}_low`] : null;
  const hi = p.band ? p.band[`${target}_high`] : null;
  return (
    <div className="min-w-0">
      <p className="text-sm font-semibold">{label}</p>
      <p className="num mt-2 flex items-baseline gap-2">
        <span className="text-[3.5rem] leading-none font-medium tracking-[-0.04em] text-led sm:text-[4.5rem]">{mmHg(value)}</span>
        <span className="text-sm text-muted-foreground">mmHg</span>
      </p>
      <BpRuler target={target} value={value} low={lo} high={hi} reference={reference} className="mt-4" />
      {lo != null && hi != null && (
        <p className="text-sm text-muted-foreground">
          Likely between <span className="num text-ink">{mmHg(lo)}</span> and <span className="num text-ink">{mmHg(hi)}</span>
        </p>
      )}
    </div>
  );
}

/** The primary model's estimate, framed like every other chart: the title states the result. */
export function ResultPanel({
  result,
  source,
}: {
  result: PredictResponse;
  selectionRule?: string;
  source?: string;
}) {
  const p = result.primary;
  const hasRef = result.true_sbp != null && result.true_dbp != null;
  const band = p.band;
  const title = hasRef
    ? `${p.display_name} estimates ${mmHg(p.sbp)} / ${mmHg(p.dbp)} mmHg. The measured value was ${mmHg(result.true_sbp!)} / ${mmHg(result.true_dbp!)}.`
    : `${p.display_name} estimates ${mmHg(p.sbp)} / ${mmHg(p.dbp)} mmHg for your signal.`;

  return (
    <section aria-label="Estimate">
      <ChartFrame
        title={title}
        subtitle={
          <>
            Systolic and diastolic estimates with their 90% range. {p.display_name} is the primary model because it had
            the lowest average error on the validation recordings.
          </>
        }
        source={
          result.input.source === "sample"
            ? (source ?? "UCI Cuff-Less Blood Pressure Estimation dataset, held-out test recording.")
            : `Your uploaded signal: the middle ${result.input.window_s} s of ${result.input.duration_s} s at ${result.input.fs_in} Hz. Not stored.`
        }
        notes={
          band ? (
            <>
              The range is sized from {band.provisional ? "test" : "validation"} errors: 90% of them were within ±
              <span className="num">{mmHg(band.half_width_sbp)}</span> mmHg systolic and ±
              <span className="num">{mmHg(band.half_width_dbp)}</span> mmHg diastolic.
              {band.provisional && " This range is provisional until validation errors are available for this model."}
              {result.mode === "demo" && " Demo model: these numbers are placeholders, not a trained estimate."}
            </>
          ) : result.mode === "demo" ? (
            "Demo model: these numbers are placeholders, not a trained estimate."
          ) : undefined
        }
      >
        <div className="px-2 sm:px-2">
          <div className="grid gap-8 sm:grid-cols-2 sm:gap-12">
            <Reading label="Systolic" target="sbp" p={p} reference={hasRef ? result.true_sbp : null} />
            <Reading label="Diastolic" target="dbp" p={p} reference={hasRef ? result.true_dbp : null} />
          </div>
          <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <Legend
              items={[
                { kind: "primary", label: `${p.display_name} estimate` },
                { kind: "primary-band", label: "Likely range (90%)" },
                ...(hasRef ? [{ kind: "truth-ring" as const, label: "Measured value" }] : []),
              ]}
            />
            {result.quality.heart_rate_bpm != null && (
              <p className="text-sm text-muted-foreground">
                Heart rate in this window: <span className="num text-ink">{Math.round(result.quality.heart_rate_bpm)}</span> bpm
              </p>
            )}
          </div>
          {result.quality.warnings.map((w) => (
            <Alert key={w} variant="caution" className="mt-4">
              <TriangleAlert />
              <p>{w}</p>
            </Alert>
          ))}
        </div>
      </ChartFrame>
    </section>
  );
}

export function ResultSkeleton() {
  return (
    <div className="frame p-6" role="status" aria-label="Estimating">
      <Skeleton className="h-6 w-3/4" />
      <Skeleton className="mt-2 h-4 w-1/2" />
      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="flex flex-col gap-4">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-16 w-36" />
            <Skeleton className="h-8 w-full" />
          </div>
        ))}
      </div>
    </div>
  );
}
