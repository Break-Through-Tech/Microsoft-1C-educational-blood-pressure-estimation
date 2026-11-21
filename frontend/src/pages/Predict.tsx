import { useState } from "react";
import { Link } from "react-router";
import { Activity } from "lucide-react";
import { OtherModels } from "@/components/OtherModels";
import { ResultPanel, ResultSkeleton } from "@/components/ResultPanel";
import { SamplePicker, useSampleParam } from "@/components/SamplePicker";
import { ChartFrame, ChartSkeleton, EmptyState, ErrorCallout, PageHeader } from "@/components/States";
import { WaveformChart, toPoints } from "@/components/WaveformChart";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/radix";
import { mmHg } from "@/lib/format";
import { useModels, usePredict, useSample, useSamples } from "@/lib/queries";
import { cn } from "@/lib/utils";

export function SampleWaveform({ id }: { id: number | null }) {
  const sample = useSample(id);
  const [view, setView] = useState<"cleaned" | "raw">("cleaned");

  if (id === null || sample.isPending) return <ChartSkeleton height={300} />;
  if (sample.isError) return <ErrorCallout title={`Couldn't load sample ${id}`} error={sample.error} onRetry={() => sample.refetch()} />;
  const d = sample.data;
  const showRaw = view === "raw" && d.raw_amplitude;
  return (
    <ChartFrame
      className={cn(sample.isPlaceholderData && "opacity-60 transition-opacity")}
      title={`Sample ${d.id}: five seconds of pulse, measured at ${mmHg(d.true_sbp)} / ${mmHg(d.true_dbp)} mmHg.`}
      subtitle={
        showRaw
          ? `Raw PPG as recorded, ${d.time_s.length} points at ${d.fs} Hz.`
          : `Cleaned PPG (filtered and scaled to −1 to 1), exactly what the model receives: ${d.time_s.length} points at ${d.fs} Hz.`
      }
      source={`${d.source}. The measured pressure comes from an arterial line in the same window.`}
      actions={
        d.raw_amplitude && (
          <ToggleGroup type="single" value={view} onValueChange={(v) => v && setView(v as "cleaned" | "raw")} aria-label="Signal view">
            <ToggleGroupItem value="cleaned">Cleaned</ToggleGroupItem>
            <ToggleGroupItem value="raw">Raw</ToggleGroupItem>
          </ToggleGroup>
        )
      }
    >
      <WaveformChart
        label={`PPG waveform for sample ${d.id}, 5 seconds`}
        yLabel={showRaw ? "Raw PPG" : "Normalized"}
        points={toPoints(d.time_s, showRaw ? d.raw_amplitude! : d.amplitude)}
      />
    </ChartFrame>
  );
}

export default function PredictPage() {
  const samples = useSamples();
  const models = useModels();
  const [id, setId] = useSampleParam(samples.data?.samples);
  const predict = usePredict();
  const current = predict.variables && "sample_id" in predict.variables ? predict.variables.sample_id : undefined;
  const resultIsForThis = current === id;
  const hasSamples = samples.isSuccess && samples.data.samples.length > 0;

  const choose = (next: number) => {
    setId(next);
    predict.reset();
  };

  return (
    <>
      <PageHeader title="Estimate from a sample">
        Each sample is five seconds of pulse from a recording the models never saw in training, with the pressure measured
        at the same time.
      </PageHeader>

      <div className="flex flex-col gap-8">
        <SamplePicker value={id} onChange={choose} />
        {hasSamples && <SampleWaveform id={id} />}
        {hasSamples && (
          <div>
            <Button size="lg" disabled={id === null || predict.isPending} onClick={() => id !== null && predict.mutate({ sample_id: id })}>
              <Activity /> {predict.isPending ? "Estimating…" : "Estimate blood pressure"}
            </Button>
          </div>
        )}
      </div>

      <div className="mt-12" aria-live="polite">
        {predict.isPending && <ResultSkeleton />}
        {predict.isError && resultIsForThis && (
          <ErrorCallout title="Couldn't estimate blood pressure" error={predict.error} onRetry={() => id !== null && predict.mutate({ sample_id: id })} />
        )}
        {predict.isSuccess && resultIsForThis && (
          <>
            <ResultPanel result={predict.data} selectionRule={models.data?.selection_rule} source={samples.data?.source} />
            <OtherModels result={predict.data} />
            <p className="mt-8 text-sm text-muted-foreground">
              Which parts of the pulse did the model rely on?{" "}
              <Link to={`/explain?sample=${id}`} className="font-semibold text-ink underline decoration-rule underline-offset-4 hover:decoration-ink">
                See the explanation for sample {id}
              </Link>
            </p>
          </>
        )}
        {predict.isIdle && hasSamples && (
          <EmptyState title="No estimate yet">
            Select Estimate blood pressure to see the primary model's reading, its likely range, and how it compares with
            the measured value.
          </EmptyState>
        )}
      </div>
    </>
  );
}
