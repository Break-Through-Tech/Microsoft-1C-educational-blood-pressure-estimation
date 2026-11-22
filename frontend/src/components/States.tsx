import type { ReactNode } from "react";
import { AlertCircle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, Skeleton } from "@/components/ui/misc";
import { cn } from "@/lib/utils";

/** An API call failed: say what happened and offer the fix. */
export function ErrorCallout({
  title,
  error,
  onRetry,
  className,
}: {
  title: string;
  error: unknown;
  onRetry?: () => void;
  className?: string;
}) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <Alert variant="caution" className={className}>
      <AlertCircle />
      <div className="flex flex-1 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="font-semibold">{title}</p>
          <p className="text-muted-foreground">{message}</p>
        </div>
        {onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry} className="self-start">
            <RotateCw /> Try again
          </Button>
        )}
      </div>
    </Alert>
  );
}

/** Nothing to show yet: say why, and what to do next. */
export function EmptyState({
  title,
  children,
  action,
  className,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("rounded-md border border-dashed border-baseline px-6 py-8", className)}>
      <p className="font-semibold">{title}</p>
      {children && <div className="measure mt-2 text-sm text-muted-foreground">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ChartSkeleton({ height = 240, className }: { height?: number; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-2", className)} role="status" aria-label="Loading chart">
      <Skeleton style={{ height }} className="w-full" />
    </div>
  );
}

export function LinesSkeleton({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-4", className)} role="status" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className="h-4" style={{ width: `${90 - i * 17}%` }} />
      ))}
    </div>
  );
}

/** Page opener: a small serif title and one short line about what the page does. */
export function PageHeader({ title, children, aside }: { title: string; children?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="mb-5">
      <h1 className="text-[1.75rem] leading-tight">{title}</h1>
      <div className="mt-1 flex items-baseline justify-between gap-6">
        {children && <div className="max-w-[70ch] text-sm text-muted-foreground">{children}</div>}
        {aside}
      </div>
    </header>
  );
}

/**
 * Every chart is framed the same way: a title that states the finding as a sentence, a subtitle that
 * says what is plotted, the chart, and a footer with the data source and notes.
 */
export function ChartFrame({
  title,
  subtitle,
  source,
  notes,
  actions,
  children,
  className,
  titleId,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  source?: ReactNode;
  notes?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  titleId?: string;
}) {
  return (
    <figure className={cn("frame flex flex-col", className)} aria-labelledby={titleId}>
      <div className="flex flex-col gap-4 px-4 pt-6 sm:flex-row sm:items-start sm:justify-between sm:px-6">
        <div className="max-w-[44rem]">
          <h2 id={titleId} className="text-lg leading-snug sm:text-[1.5rem]">
            {title}
          </h2>
          {subtitle && <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>}
        </div>
        {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
      </div>
      <div className="px-2 pt-4 pb-4 sm:px-4">{children}</div>
      {(source || notes) && (
        <figcaption className="flex flex-col gap-1 border-t border-rule px-4 py-4 text-xs leading-relaxed text-muted-foreground sm:px-6">
          {source && (
            <p>
              <span className="font-semibold text-ink">Source:</span> {source}
            </p>
          )}
          {notes && (
            <div>
              <span className="font-semibold text-ink">Note:</span> {notes}
            </div>
          )}
        </figcaption>
      )}
    </figure>
  );
}

/** Legend swatches for the fixed chart mapping. */
export type LegendKind = "truth-line" | "truth-ring" | "primary" | "primary-band" | "other" | "baseline";

export function Swatch({ kind, color }: { kind: LegendKind; color?: string }) {
  switch (kind) {
    case "truth-line":
      return <span aria-hidden className="inline-block h-0.5 w-5 bg-truth" />;
    case "truth-ring":
      return <span aria-hidden className="inline-block size-3 rounded-full border-2 border-truth bg-surface" />;
    case "primary":
      return <span aria-hidden className="inline-block h-3 w-[3px] bg-led" />;
    case "primary-band":
      return <span aria-hidden className="inline-block h-3 w-5 rounded-[2px] border-x-2 border-led bg-led-soft" />;
    case "other":
      return <span aria-hidden className="inline-block size-3 rounded-[2px]" style={{ background: color }} />;
    case "baseline":
      return <span aria-hidden className="inline-block h-3 w-5 rounded-[2px] border border-dashed border-baseline" />;
  }
}

export function Legend({ items, className }: { items: { kind: LegendKind; label: string; color?: string }[]; className?: string }) {
  return (
    <ul className={cn("flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground", className)}>
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-2">
          <Swatch kind={i.kind} color={i.color} />
          {i.label}
        </li>
      ))}
    </ul>
  );
}
