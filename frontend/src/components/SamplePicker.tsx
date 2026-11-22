import { useCallback, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useQueries } from "@tanstack/react-query";
import { api, type SampleSummary } from "@/api";
import { EmptyState, ErrorCallout } from "@/components/States";
import { Label, Skeleton } from "@/components/ui/misc";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/radix";
import { mmHg, recordingLabel } from "@/lib/format";
import { useSamples } from "@/lib/queries";
import { useChartColors } from "@/lib/tokens";
import { cn } from "@/lib/utils";

/** The selected sample lives in the URL (?sample=12), so Predict and Explain share it and links work. */
export function useSampleParam(samples: SampleSummary[] | undefined): [number | null, (id: number) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("sample");
  const asked = raw !== null && /^\d+$/.test(raw) ? Number(raw) : null;
  let id: number | null = null;
  if (samples && samples.length > 0) {
    id = asked !== null && samples.some((s) => s.id === asked) ? asked : samples[0].id;
  }
  const setId = useCallback(
    (next: number) =>
      setParams(
        (p) => {
          const q = new URLSearchParams(p);
          q.set("sample", String(next));
          return q;
        },
        { replace: true },
      ),
    [setParams],
  );
  return [id, setId];
}

/**
 * How people name a sample: "Recording 12", plus a 1-based window number when that recording has
 * more than one window in the list ("Recording 12, window 2"). Never the 0-based API id.
 */
export function useSampleNames(list: SampleSummary[] | undefined): Map<number, string> {
  return useMemo(() => {
    const names = new Map<number, string>();
    if (!list) return names;
    const total = new Map<number, number>();
    for (const s of list) total.set(s.recording_id, (total.get(s.recording_id) ?? 0) + 1);
    const seen = new Map<number, number>();
    for (const s of list) {
      const k = (seen.get(s.recording_id) ?? 0) + 1;
      seen.set(s.recording_id, k);
      names.set(s.id, total.get(s.recording_id)! > 1 ? `${recordingLabel(s.recording_id)}, window ${k}` : recordingLabel(s.recording_id));
    }
    return names;
  }, [list]);
}

const STRIP_SIZE = 8;

/** Up to `n` samples from distinct recordings, starting at a random offset (or the front when seed is 0). */
function pickSet(list: SampleSummary[], n: number, random: boolean, avoid: Set<number> = new Set()): number[] {
  const order = random ? [...list].sort(() => Math.random() - 0.5) : list;
  const out: SampleSummary[] = [];
  const recordings = new Set<number>();
  for (const pass of [0, 1, 2]) {
    for (const s of order) {
      if (out.length >= n) break;
      if (out.includes(s)) continue;
      if (pass === 0 && (recordings.has(s.recording_id) || avoid.has(s.id))) continue;
      if (pass === 1 && recordings.has(s.recording_id)) continue;
      out.push(s);
      recordings.add(s.recording_id);
    }
  }
  return out.map((s) => s.id);
}

function Sparkline({ values, selected }: { values: number[] | undefined; selected: boolean }) {
  const c = useChartColors();
  if (!values) return <Skeleton className="h-8 w-full rounded-[2px]" />;
  // every 3rd point is plenty at this size
  const pts: string[] = [];
  const n = values.length;
  for (let i = 0; i < n; i += 3) pts.push(`${((i / (n - 1)) * 100).toFixed(1)},${(16 - values[i] * 14).toFixed(1)}`);
  return (
    <svg viewBox="0 0 100 32" preserveAspectRatio="none" className="block h-8 w-full" aria-hidden>
      <polyline points={pts.join(" ")} fill="none" stroke={selected ? c.primary : c.ink} strokeWidth={1.25} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

/** Which thumbnails are on show, and a way to load another set (distinct recordings, avoiding the current set). */
export function useSampleSet(value: number | null): [number[], () => void] {
  const samples = useSamples();
  const list = samples.data?.samples;
  const [set, setSet] = useState<number[] | null>(null);
  // first view: the first recordings in the list, always including the one in the URL
  const shown = useMemo(() => {
    if (set) return set;
    if (!list?.length) return [];
    const first = pickSet(list, STRIP_SIZE, false);
    if (value !== null && !first.includes(value)) first[first.length - 1] = value;
    return first;
  }, [set, list, value]);
  const shuffle = useCallback(() => list && setSet(pickSet(list, STRIP_SIZE, true, new Set(shown))), [list, shown]);
  return [shown, shuffle];
}

export function ShuffleLink({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="shrink-0 rounded-[2px] text-sm text-ink underline decoration-rule underline-offset-4 hover:decoration-ink"
    >
      Shuffle
    </button>
  );
}

/** A row of small pulse thumbnails, one per recording. The selected one has an accent border. */
export function SampleStrip({ shown, value, onChange }: { shown: number[]; value: number | null; onChange: (id: number) => void }) {
  const samples = useSamples();
  const list = samples.data?.samples;
  const names = useSampleNames(list);
  const details = useQueries({
    queries: shown.map((id) => ({ queryKey: ["sample", id], queryFn: () => api.sample(id), staleTime: 5 * 60_000 })),
  });

  if (samples.isPending)
    return (
      <div className="flex gap-2 overflow-hidden" role="status" aria-label="Loading samples">
        {Array.from({ length: STRIP_SIZE }, (_, i) => (
          <Skeleton key={i} className="h-[66px] w-[136px] shrink-0" />
        ))}
      </div>
    );
  if (samples.isError)
    return <ErrorCallout title="Couldn't load the sample list" error={samples.error} onRetry={() => samples.refetch()} />;
  if (!list || list.length === 0)
    return (
      <EmptyState title="No samples available">
        The server has no held-out recordings to show. Build the data cache and run the evaluation script, then reload
        this page.
      </EmptyState>
    );

  const byId = new Map(list.map((s) => [s.id, s]));
  return (
    <ul
      className="-mx-1 flex gap-2 overflow-x-auto px-1 py-1 lg:grid lg:grid-cols-8 lg:overflow-visible"
      aria-label="Recordings"
    >
      {shown.map((id, i) => {
        const s = byId.get(id);
        if (!s) return null;
        const selected = id === value;
        const rec = names.get(id)?.replace(/, window \d+$/, "");
        const bp = `${mmHg(s.true_sbp)}/${mmHg(s.true_dbp)}`;
        const text = `${rec}, measured ${bp} mmHg`;
        return (
          <li key={id} className="w-[136px] shrink-0 lg:w-auto lg:min-w-0">
            <button
              type="button"
              aria-pressed={selected}
              title={text}
              onClick={() => onChange(id)}
              className={cn(
                "flex w-full flex-col gap-1.5 rounded-[4px] border bg-surface px-1.5 pt-2 pb-1.5 text-left transition-colors",
                selected ? "border-led outline outline-1 -outline-offset-2 outline-led" : "border-rule-soft hover:border-rule",
              )}
            >
              <Sparkline values={details[i]?.data?.amplitude} selected={selected} />
              <span className="num flex flex-col text-[0.75rem] leading-snug">
                <span className={cn("truncate", selected ? "font-medium text-ink" : "text-muted-foreground")}>{rec}</span>
                <span className={selected ? "text-ink" : "text-muted-foreground"}>{bp}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Dropdown version, for pages with less room (Explain). */
export function SamplePicker({ value, onChange }: { value: number | null; onChange: (id: number) => void }) {
  const samples = useSamples();
  const names = useSampleNames(samples.data?.samples);

  if (samples.isPending)
    return (
      <div className="flex flex-col gap-2" role="status" aria-label="Loading samples">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-10 w-full max-w-md" />
      </div>
    );
  if (samples.isError)
    return <ErrorCallout title="Couldn't load the sample list" error={samples.error} onRetry={() => samples.refetch()} />;
  const list = samples.data.samples;
  if (list.length === 0)
    return (
      <EmptyState title="No samples available">
        The server has no held-out recordings to show. Build the data cache and run the evaluation script, then reload
        this page.
      </EmptyState>
    );

  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor="sample-picker">Recording</Label>
      <Select value={value === null ? undefined : String(value)} onValueChange={(v) => onChange(Number(v))}>
        <SelectTrigger id="sample-picker" className="w-full sm:w-80">
          <SelectValue placeholder="Pick a recording" />
        </SelectTrigger>
        <SelectContent>
          {list.map((s) => (
            <SelectItem key={s.id} value={String(s.id)}>
              <span className="num">
                {names.get(s.id)}
                <span className="ml-4 text-muted-foreground">
                  {mmHg(s.true_sbp)}/{mmHg(s.true_dbp)}
                </span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
