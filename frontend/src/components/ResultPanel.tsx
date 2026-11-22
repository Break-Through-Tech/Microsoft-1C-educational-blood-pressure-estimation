import type { ReactNode } from "react";
import { Link } from "react-router";
import { Info, TriangleAlert } from "lucide-react";
import type { PredictResponse } from "@/api";
import { Alert, Skeleton } from "@/components/ui/misc";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/radix";
import { mmHg, signed } from "@/lib/format";

function Row({ term, children, hint }: { term: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="grid grid-cols-[6.5rem_1fr] items-baseline gap-x-4 border-t border-rule-soft py-2.5">
      <dt className="text-sm text-muted-foreground">{term}</dt>
      <dd className="num text-base text-ink">
        {children}
        {hint && <span className="block text-xs text-muted-foreground">{hint}</span>}
      </dd>
    </div>
  );
}

export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" aria-label={label} className="inline-flex rounded-full text-muted-foreground hover:text-ink">
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{children}</TooltipContent>
    </Tooltip>
  );
}

/**
 * The primary model's estimate: large tabular figures, then measured value, error, error band and
 * heart rate as a short list. Works for dataset samples (with a measured value) and user signals.
 */
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
  const isSample = result.input.source === "sample";

  return (
    <section aria-label="Estimate" className="flex flex-col">
      <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
        {p.display_name}
        <InfoTip label="Why this model">
          The primary model: it had the lowest average error on the validation recordings, which were not used to
          train it.
        </InfoTip>
      </p>
      <p className="num mt-1 flex items-baseline gap-2 whitespace-nowrap" aria-label={`Estimate ${mmHg(p.sbp)} over ${mmHg(p.dbp)} mmHg`}>
        <span className="text-[3.75rem] leading-none font-light tracking-[-0.045em] text-led">
          {mmHg(p.sbp)}
          <span className="mx-1 text-rule">/</span>
          {mmHg(p.dbp)}
        </span>
        <span className="text-sm text-muted-foreground">mmHg</span>
      </p>
      <p className="mt-1 mb-4 text-xs text-muted-foreground">Systolic / diastolic</p>

      <dl>
        {hasRef && (
          <>
            <Row term="Measured">
              {mmHg(result.true_sbp!)} / {mmHg(result.true_dbp!)}
            </Row>
            <Row term="Error">
              {signed(p.sbp - result.true_sbp!)} / {signed(p.dbp - result.true_dbp!)} <span className="text-sm text-muted-foreground">mmHg</span>
            </Row>
          </>
        )}
        {band && (
          <Row
            term="Error band"
            hint={
              <>
                {mmHg(band.sbp_low)}–{mmHg(band.sbp_high)} / {mmHg(band.dbp_low)}–{mmHg(band.dbp_high)}
                <br />
                Holds 90% of {band.provisional ? "test" : "validation"} errors
              </>
            }
          >
            ±{mmHg(band.half_width_sbp)} / ±{mmHg(band.half_width_dbp)}
          </Row>
        )}
        {result.quality.heart_rate_bpm != null && (
          <Row term="Heart rate">
            {Math.round(result.quality.heart_rate_bpm)} <span className="text-sm text-muted-foreground">bpm</span>
          </Row>
        )}
      </dl>

      <div className="mt-2 flex flex-col gap-1 border-t border-rule-soft pt-3 text-xs leading-relaxed text-muted-foreground">
        {band?.provisional && <p>Provisional band: this model has no validation errors yet.</p>}
        {result.mode === "demo" && <p>Demo model: placeholder numbers, not a trained estimate.</p>}
        {!isSample && (
          <p>
            Your signal: the middle {result.input.window_s} s of {result.input.duration_s.toFixed(1)} s at{" "}
            {result.input.fs_in.toFixed(1)} Hz, resampled to 125 Hz. Not stored.
          </p>
        )}
        {isSample && source && result.mode !== "demo" && <p>{source}</p>}
        {isSample && result.input.sample_id != null && (
          <Link
            to={`/explain?sample=${result.input.sample_id}`}
            className="self-start text-sm text-ink underline decoration-rule underline-offset-4 hover:decoration-ink"
          >
            See what the model relied on
          </Link>
        )}
      </div>

      {result.quality.warnings.map((w) => (
        <Alert key={w} variant="caution" className="mt-4">
          <TriangleAlert />
          <p>{w}</p>
        </Alert>
      ))}
    </section>
  );
}

export function ResultSkeleton() {
  return (
    <div className="flex flex-col gap-3" role="status" aria-label="Estimating">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-14 w-52" />
      <Skeleton className="mt-4 h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
    </div>
  );
}
