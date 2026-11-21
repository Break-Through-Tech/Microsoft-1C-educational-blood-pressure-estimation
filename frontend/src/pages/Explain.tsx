import { useState } from "react";
import { useSearchParams } from "react-router";
import type { ReactNode } from "react";
import type { ExplainResponse, ModelInfo, SegmentImportance } from "@/api";
import { FeatureImportanceChart } from "@/components/FeatureImportanceChart";
import { SamplePicker, useSampleParam } from "@/components/SamplePicker";
import { ChartFrame, ChartSkeleton, EmptyState, ErrorCallout, Legend, PageHeader } from "@/components/States";
import { seriesColor, useChartColors } from "@/lib/tokens";
import { WaveformChart, toPoints } from "@/components/WaveformChart";
import { Label, Skeleton } from "@/components/ui/misc";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, ToggleGroup, ToggleGroupItem } from "@/components/ui/radix";
import { pct } from "@/lib/format";
import { useExplain, useModels, useSample, useSamples } from "@/lib/queries";

function ModelSelect({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor="model-select">Model</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id="model-select" className="w-full sm:w-72">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {models.map((m) => (
            <SelectItem key={m.name} value={m.name}>
              {m.display_name}
              {m.is_primary ? " (primary)" : ""}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function Saliency({ e, sampleId, target, modelName, actions }: { e: ExplainResponse; sampleId: number; target: "sbp" | "dbp"; modelName: string; actions: ReactNode }) {
  const sample = useSample(sampleId);
  const segs = e.segments ?? [];
  if (sample.isPending) return <ChartSkeleton height={300} />;
  if (sample.isError) return <ErrorCallout title={`Couldn't load sample ${sampleId}`} error={sample.error} onRetry={() => sample.refetch()} />;
  const total = segs.reduce((a, s) => a + s[target], 0) || 1;
  const top = [...segs].sort((a, b) => b[target] - a[target]).slice(0, 2);
  const shares = segs.map((s) => s[target] / total);
  const even = shares.length > 0 && Math.max(...shares) - Math.min(...shares) < 0.005;
  const which = target === "sbp" ? "systolic" : "diastolic";
  const span = (s: SegmentImportance) => `${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)} s`;
  const title = even
    ? `For sample ${sampleId}, the ${which} estimate depends evenly on every part of the pulse.`
    : `For sample ${sampleId}, the ${which} estimate leaned most on ${top.map(span).join(" and ")}.`;
  return (
    <ChartFrame
      title={title}
      subtitle={`${modelName}: cleaned PPG with each half-second shaded by how strongly small changes there would move the ${which} estimate, relative to the rest of this sample.`}
      source={`Gradient saliency computed on sample ${sampleId} from the held-out test recordings.`}
      notes="Shading shows where the estimate is sensitive, not why. It isn't evidence of a physiological cause."
      actions={actions}
    >
      <WaveformChart
        height={280}
        label={`Cleaned PPG for sample ${sampleId} with influential time regions shaded`}
        yLabel="Normalized"
        points={toPoints(sample.data.time_s, sample.data.amplitude)}
        segments={segs}
        target={target}
      />
      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 px-2 text-sm text-muted-foreground">
        <Legend items={[{ kind: "truth-line", label: "PPG signal" }]} />
        <span className="flex items-center gap-2">
          <span aria-hidden className="inline-block h-3 w-16 rounded-[2px]" style={{ background: "linear-gradient(90deg, transparent, var(--led))", border: "1px solid var(--rule)" }} />
          Less to more influence on {modelName}
        </span>
        {!even && (
          <span>
            Top regions: {top.map((s) => <span key={s.start_s} className="num text-ink">{span(s)} ({pct(s[target] / total)}) </span>)}
          </span>
        )}
      </div>
      <table className="sr-only">
        <caption>Share of influence per time segment</caption>
        <thead>
          <tr>
            <th>Segment</th>
            <th>Share</th>
          </tr>
        </thead>
        <tbody>
          {segs.map((s) => (
            <tr key={s.start_s}>
              <td>{s.start_s.toFixed(1)} to {s.end_s.toFixed(1)} seconds</td>
              <td>{pct(s[target] / total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ChartFrame>
  );
}

export default function ExplainPage() {
  const models = useModels();
  const samples = useSamples();
  const [params, setParams] = useSearchParams();
  const [sampleId, setSampleId] = useSampleParam(samples.data?.samples);
  const [target, setTarget] = useState<"sbp" | "dbp">("sbp");

  const list = models.data?.models ?? [];
  const asked = params.get("model");
  const model = list.find((m) => m.name === asked)?.name ?? models.data?.primary_model ?? null;
  const setModel = (m: string) =>
    setParams(
      (p) => {
        const q = new URLSearchParams(p);
        q.set("model", m);
        return q;
      },
      { replace: true },
    );
  const explain = useExplain(model, sampleId);
  const info = list.find((m) => m.name === model);
  const c = useChartColors();
  const targetToggle = (
    <ToggleGroup type="single" value={target} onValueChange={(v) => v && setTarget(v as "sbp" | "dbp")} aria-label="Pressure">
      <ToggleGroupItem value="sbp">Systolic</ToggleGroupItem>
      <ToggleGroupItem value="dbp">Diastolic</ToggleGroupItem>
    </ToggleGroup>
  );

  return (
    <>
      <PageHeader title="What the model paid attention to">
        Pick a model and a sample to see which parts of the input moved the estimate most.
      </PageHeader>

      <div className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-start lg:gap-8">
        {models.isPending && (
          <div className="flex flex-col gap-2" role="status" aria-label="Loading models">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-10 w-72" />
          </div>
        )}
        {models.isError && <ErrorCallout title="Couldn't load the model list" error={models.error} onRetry={() => models.refetch()} />}
        {models.isSuccess && model && <ModelSelect models={list} value={model} onChange={setModel} />}
        <div className="flex-1">
          <SamplePicker value={sampleId} onChange={setSampleId} />
        </div>
      </div>

      {models.isSuccess && list.length === 0 && (
        <EmptyState title="No models loaded">The server has no models to explain. Run the evaluation script, then reload this page.</EmptyState>
      )}

      {model !== null && sampleId !== null && (
        <section aria-live="polite">
          {explain.isPending && <ChartSkeleton height={260} />}
          {explain.isError && (
            <ErrorCallout title="Couldn't build the explanation" error={explain.error} onRetry={() => explain.refetch()} />
          )}
          {explain.isSuccess && (
            <div className={explain.isPlaceholderData ? "opacity-60 transition-opacity" : undefined}>
              {explain.data.kind === "saliency" && (
                <Saliency e={explain.data} sampleId={sampleId} target={target} modelName={info?.display_name ?? model} actions={targetToggle} />
              )}
              {explain.data.kind === "feature_importance" && (explain.data.features?.length ?? 0) > 0 && (
                <ChartFrame
                  title={`${info?.display_name ?? model} relies most on ${[...explain.data.features!].sort((a, b) => b[target] - a[target])[0].name} for ${target === "sbp" ? "systolic" : "diastolic"} pressure.`}
                  subtitle="How much each input feature contributes to the estimate. Longer bars matter more."
                  source={`Computed from ${info?.display_name ?? model} (${explain.data.method}).`}
                  actions={targetToggle}
                >
                  <FeatureImportanceChart
                    features={explain.data.features!}
                    target={target}
                    color={info ? seriesColor(c, info, Math.max(0, list.filter((m) => !m.is_primary && !m.is_baseline).indexOf(info))) : undefined}
                  />
                </ChartFrame>
              )}
              {explain.data.kind === "feature_importance" && (explain.data.features?.length ?? 0) === 0 && (
                <EmptyState title="No feature importances returned">This model didn't report any feature importances. Pick another model.</EmptyState>
              )}
              {explain.data.kind === "none" && (
                <EmptyState title="Nothing to explain for this model">
                  {info?.is_baseline
                    ? "The baseline predicts the same value for every signal, so no part of the pulse affects it. Pick another model."
                    : "This model doesn't produce explanations yet. Pick another model."}
                </EmptyState>
              )}
            </div>
          )}
        </section>
      )}
    </>
  );
}
