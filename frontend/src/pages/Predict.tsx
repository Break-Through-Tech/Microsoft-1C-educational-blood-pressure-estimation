import { useEffect, useState } from "react";
import { OtherModels } from "@/components/OtherModels";
import { PulseStrip } from "@/components/PulseStrip";
import { InfoTip, ResultPanel, ResultSkeleton } from "@/components/ResultPanel";
import { SampleStrip, ShuffleLink, useSampleNames, useSampleParam, useSampleSet } from "@/components/SamplePicker";
import { ChartSkeleton, ErrorCallout, PageHeader } from "@/components/States";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/radix";
import { usePredict, useSample, useSamples } from "@/lib/queries";
import { cn } from "@/lib/utils";

function SampleWaveform({ id, name }: { id: number | null; name?: string }) {
  const sample = useSample(id);
  const [view, setView] = useState<"cleaned" | "raw">("cleaned");

  if (id === null || sample.isPending) return <ChartSkeleton height={340} />;
  if (sample.isError)
    return <ErrorCallout title={`Couldn't load ${name ?? "this recording"}`} error={sample.error} onRetry={() => sample.refetch()} />;
  const d = sample.data;
  const showRaw = view === "raw" && d.raw_amplitude != null;
  const nBeats = d.beats?.peaks_s.length ?? 0;

  return (
    <figure className={cn("min-w-0", sample.isPlaceholderData && "opacity-60 transition-opacity")}>
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <figcaption className="flex items-center gap-1.5">
          <span className="text-base font-semibold">Pulse</span>
          {name && <span className="text-sm text-muted-foreground">{name}</span>}
          <InfoTip label="About this signal">
            {showRaw ? (
              <>
                Raw PPG as recorded: {d.time_s.length} points at {d.fs} Hz, before filtering.
              </>
            ) : (
              <>
                Cleaned PPG, exactly what the model receives: {d.time_s.length} points at {d.fs} Hz, band-pass filtered
                (0.5–5 Hz) and scaled to −1 to 1. Beats come from the pipeline's peak detector.
              </>
            )}{" "}
            Measured pressure comes from an arterial line in the same window. Source: {d.source}.
          </InfoTip>
        </figcaption>
        {d.raw_amplitude && (
          <ToggleGroup type="single" value={view} onValueChange={(v) => v && setView(v as "cleaned" | "raw")} aria-label="Signal view">
            <ToggleGroupItem value="cleaned">Cleaned</ToggleGroupItem>
            <ToggleGroupItem value="raw">Raw</ToggleGroupItem>
          </ToggleGroup>
        )}
      </div>
      <PulseStrip
        time={d.time_s}
        values={showRaw ? d.raw_amplitude! : d.amplitude}
        fs={d.fs}
        beats={d.beats}
        landmarks={!showRaw}
        drawKey={d.id}
        height={340}
        label={`PPG pulse for ${name ?? "the selected recording"}, 5 seconds, ${nBeats} beats detected`}
      />
      {nBeats > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          <span className="num">{nBeats}</span> beats detected (circles).
          {!showRaw && d.beats?.representative && " The shaded beat is labelled."}
        </p>
      )}
    </figure>
  );
}

export default function PredictPage() {
  const samples = useSamples();
  const [id, setId] = useSampleParam(samples.data?.samples);
  const names = useSampleNames(samples.data?.samples);
  const [shown, shuffle] = useSampleSet(id);
  const predict = usePredict();
  const { mutate, reset } = predict;
  const current = predict.variables && "sample_id" in predict.variables ? predict.variables.sample_id : undefined;
  const resultIsForThis = current === id;
  const hasSamples = samples.isSuccess && samples.data.samples.length > 0;

  // picking a recording estimates it straight away, so the result sits beside its pulse
  useEffect(() => {
    if (id === null) return;
    reset();
    mutate({ sample_id: id });
  }, [id, mutate, reset]);

  const name = id !== null ? names.get(id) : undefined;

  return (
    <>
      <PageHeader title="Predict" aside={hasSamples && <ShuffleLink onClick={shuffle} />}>
        Pick a recording to estimate its blood pressure from five seconds of pulse.
      </PageHeader>

      <SampleStrip shown={shown} value={id} onChange={setId} />

      {hasSamples && (
        <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,1fr)_330px] lg:gap-12">
          <SampleWaveform id={id} name={name} />
          <div className="lg:border-l lg:border-rule-soft lg:pl-10" aria-live="polite">
            {(predict.isPending || predict.isIdle) && <ResultSkeleton />}
            {predict.isError && resultIsForThis && (
              <ErrorCallout title="Couldn't estimate blood pressure" error={predict.error} onRetry={() => id !== null && mutate({ sample_id: id })} />
            )}
            {predict.isSuccess && resultIsForThis && <ResultPanel result={predict.data} source={samples.data?.source} />}
          </div>
        </div>
      )}

      {predict.isSuccess && resultIsForThis && <OtherModels result={predict.data} />}
    </>
  );
}
