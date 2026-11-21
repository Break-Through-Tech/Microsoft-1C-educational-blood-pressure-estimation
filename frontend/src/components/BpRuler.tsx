import { mmHg } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * A horizontal mmHg scale: the estimate is a tick, the 90% range a span, the true (reference) value a
 * ring. Fixed mapping: primary = accent, other models = muted tone, baseline = gray dashed,
 * true value = neutral dark. Plain numbers only: no colored zones, no clinical categories.
 */
export const SCALES = {
  sbp: { min: 60, max: 200, step: 20 },
  dbp: { min: 30, max: 130, step: 20 },
} as const;

type Variant = "primary" | "baseline" | "other";

interface Props {
  target: "sbp" | "dbp";
  value: number;
  low?: number | null;
  high?: number | null;
  reference?: number | null;
  variant?: Variant;
  /** muted tone for variant "other" */
  color?: string;
  size?: "full" | "mini";
  className?: string;
}

export function BpRuler({ target, value, low, high, reference, variant = "primary", color, size = "full", className }: Props) {
  const { min, max, step } = SCALES[target];
  const pos = (v: number) => ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * 100;
  const hasBand = low != null && high != null;
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(t);

  const label =
    `${target === "sbp" ? "Systolic" : "Diastolic"} estimate ${mmHg(value)} mmHg` +
    (hasBand ? `, likely between ${mmHg(low!)} and ${mmHg(high!)}` : "") +
    (reference != null ? `, true value ${mmHg(reference)}` : "");

  const full = size === "full";
  const bandStyle =
    variant === "primary"
      ? "bg-led-soft border-x-2 border-led"
      : variant === "baseline"
        ? "border border-dashed border-baseline"
        : "border-x-2";
  const tickStyle = variant === "primary" ? "bg-led" : variant === "baseline" ? "border-l-2 border-dashed border-baseline" : "";
  const otherBand = variant === "other" && color ? { background: `${color}33`, borderColor: color } : {};
  const otherTick = variant === "other" && color ? { background: color } : {};
  const origin = hasBand && pos(high!) > pos(low!) ? ((pos(value) - pos(low!)) / (pos(high!) - pos(low!))) * 100 : 50;

  return (
    <div role="img" aria-label={label} className={cn("relative w-full select-none", full ? "px-4 pt-2 pb-6" : "py-1", className)}>
      <div className={cn("relative", full ? "h-8" : "h-4")}>
        <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-rule" />
        {full &&
          ticks.map((t) => (
            <div key={t} className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-rule" style={{ left: `${pos(t)}%` }} />
          ))}
        {hasBand && (
          <div
            className={cn("absolute inset-y-0 rounded-[2px]", bandStyle)}
            style={{
              left: `${pos(low!)}%`,
              width: `${pos(high!) - pos(low!)}%`,
              ...otherBand,
              // the range grows out of the estimate, so it reads as part of the answer
              transformOrigin: `${origin}% 50%`,
              animation: "band-grow .6s cubic-bezier(.2,.7,.2,1) .1s both",
            }}
          />
        )}
        <div
          className={cn("absolute -translate-x-1/2", tickStyle, full ? "-inset-y-1 w-[3px]" : "-inset-y-0.5 w-0.5", variant === "baseline" && "w-0")}
          style={{ left: `${pos(value)}%`, ...otherTick }}
        />
        {reference != null && (
          <div
            className={cn(
              "absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-truth bg-surface",
              full ? "size-3.5 border-2" : "size-2.5 border-[1.5px]",
            )}
            style={{ left: `${pos(reference)}%` }}
          />
        )}
      </div>
      {full && (
        <div className="num relative mt-2 h-4 text-xs text-muted-foreground" aria-hidden>
          {ticks.map((t, i) => (
            <span key={t} className={cn("absolute -translate-x-1/2", i % 2 === 1 && "max-sm:hidden")} style={{ left: `${pos(t)}%` }}>
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
