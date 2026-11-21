import { useCallback } from "react";
import { useSearchParams } from "react-router";
import { Shuffle } from "lucide-react";
import type { SampleSummary } from "@/api";
import { EmptyState, ErrorCallout } from "@/components/States";
import { Button } from "@/components/ui/button";
import { Label, Skeleton } from "@/components/ui/misc";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/radix";
import { mmHg } from "@/lib/format";
import { useSamples } from "@/lib/queries";

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

export function SamplePicker({ value, onChange }: { value: number | null; onChange: (id: number) => void }) {
  const samples = useSamples();

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

  const random = () => {
    const pool = list.filter((s) => s.id !== value);
    const pick = pool[Math.floor(Math.random() * pool.length)] ?? list[0];
    onChange(pick.id);
  };

  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor="sample-picker">Choose a sample</Label>
      <div className="flex flex-wrap items-center gap-2">
        <Select value={value === null ? undefined : String(value)} onValueChange={(v) => onChange(Number(v))}>
          <SelectTrigger id="sample-picker" className="w-full sm:w-80">
            <SelectValue placeholder="Pick a sample" />
          </SelectTrigger>
          <SelectContent>
            {list.map((s) => (
              <SelectItem key={s.id} value={String(s.id)}>
                <span className="num">
                  Sample {s.id}
                  <span className="ml-4 text-muted-foreground">
                    measured {mmHg(s.true_sbp)} / {mmHg(s.true_dbp)}
                  </span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="outline" onClick={random}>
          <Shuffle /> Random sample
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">{samples.data.source}</p>
    </div>
  );
}
