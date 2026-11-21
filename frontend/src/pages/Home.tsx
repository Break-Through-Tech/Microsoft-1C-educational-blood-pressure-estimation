import { Link } from "react-router";
import type { MetricsResponse } from "@/api";
import { HeroTrace, beatStats } from "@/components/HeroTrace";
import { PpgDiagram } from "@/components/PulseTrace";
import { ChartFrame, ChartSkeleton, ErrorCallout, Legend } from "@/components/States";
import { Button } from "@/components/ui/button";
import { err1 } from "@/lib/format";
import { useMetrics, useSample, useSamples } from "@/lib/queries";

/** The hero: a real recording from the held-out test set. */
function RealPulse() {
  const samples = useSamples();
  const firstId = samples.data?.samples[0]?.id ?? null;
  const sample = useSample(firstId);

  if (samples.isError) return <ErrorCallout title="Couldn't load a recording" error={samples.error} onRetry={() => samples.refetch()} />;
  if (sample.isError) return <ErrorCallout title="Couldn't load a recording" error={sample.error} onRetry={() => sample.refetch()} />;
  if (samples.isSuccess && firstId === null) return null;
  if (!sample.data) return <ChartSkeleton height={340} />;

  const { peaks, bpm } = beatStats(sample.data);
  const title =
    bpm !== null
      ? `Each peak is one heartbeat. This recording has ${peaks.length} beats in 5 seconds, about ${Math.round(bpm)} per minute.`
      : "Five seconds of a real pulse recording.";
  return (
    <ChartFrame
      titleId="pulse-title"
      title={title}
      subtitle="Light reflected from a fingertip, cleaned and scaled to the range −1 to 1. The trace plays at the speed it was recorded."
      source={`${sample.data.source}, sample ${sample.data.id}.`}
    >
      <div className="px-2">
        <HeroTrace sample={sample.data} />
      </div>
    </ChartFrame>
  );
}

function maeRows(d: MetricsResponse) {
  const pick = (model: string, t: "sbp" | "dbp") => d.rows.find((r) => r.model === model && r.split === "test" && r.target === t);
  const baseline = d.rows.find((r) => r.is_baseline)?.model;
  return {
    primary: { sbp: pick(d.primary_model, "sbp"), dbp: pick(d.primary_model, "dbp") },
    baseline: baseline ? { sbp: pick(baseline, "sbp"), dbp: pick(baseline, "dbp") } : null,
  };
}

function MaeBar({ value, max, kind }: { value: number; max: number; kind: "primary" | "baseline" }) {
  return (
    <div className="flex items-center gap-4">
      <div className="relative h-6 flex-1">
        <div
          className={kind === "primary" ? "h-full rounded-[2px] bg-led" : "h-full rounded-[2px] border-[1.5px] border-dashed border-baseline"}
          style={{ width: `${(value / max) * 100}%` }}
        />
      </div>
      <span className="num w-24 shrink-0 text-sm">{err1(value)} mmHg</span>
    </div>
  );
}

/** The headline result as a small chart: model error next to the guess-the-average baseline. */
function HeadlineResult() {
  const metrics = useMetrics();
  if (metrics.isPending) return <ChartSkeleton height={260} />;
  if (metrics.isError) return <ErrorCallout title="Couldn't load the results" error={metrics.error} onRetry={() => metrics.refetch()} />;
  const d = metrics.data;
  const { primary, baseline } = maeRows(d);
  if (!primary.sbp || !primary.dbp) return null;
  const name = primary.sbp.display_name;
  const max = Math.max(primary.sbp.mae, primary.dbp.mae, baseline?.sbp?.mae ?? 0, baseline?.dbp?.mae ?? 0) * 1.1;

  return (
    <ChartFrame
      titleId="result-title"
      title={`On recordings it never saw, ${name} misses systolic pressure by ${err1(primary.sbp.mae)} mmHg on average.`}
      subtitle="Mean absolute error on the held-out test recordings. Lower is better."
      source={`UCI Cuff-Less Blood Pressure Estimation dataset, test split (${primary.sbp.n.toLocaleString()} five-second windows).`}
      notes={
        <>
          The baseline guesses the training average for everyone, so the gap between the bars is what the model learned
          from the pulse. A validated cuff must keep its average error within 5 mmHg with a spread under 8 mmHg (the AAMI
          standard). This prototype does not meet that.{d.mode === "demo" && " These are demo numbers."}
        </>
      }
    >
      <div className="flex flex-col gap-6 px-2 py-2">
        {(["sbp", "dbp"] as const).map((t) => (
          <div key={t} className="grid gap-2 sm:grid-cols-[120px_1fr] sm:items-center sm:gap-6">
            <p className="text-sm font-semibold">{t === "sbp" ? "Systolic" : "Diastolic"}</p>
            <div className="flex flex-col gap-2">
              <MaeBar value={primary[t]!.mae} max={max} kind="primary" />
              {baseline?.[t] && <MaeBar value={baseline[t]!.mae} max={max} kind="baseline" />}
            </div>
          </div>
        ))}
        <Legend
          items={[
            { kind: "other", label: name, color: "var(--led)" },
            ...(baseline?.sbp ? [{ kind: "baseline" as const, label: baseline.sbp.display_name }] : []),
          ]}
        />
      </div>
    </ChartFrame>
  );
}

const STEPS = [
  { title: "Light goes in", body: "A green LED shines into the skin of a fingertip or wrist." },
  {
    title: "Blood changes how much comes back",
    body: "With each heartbeat the vessels fill slightly, absorb more light, and reflect less. A sensor beside the LED records that as a wave.",
  },
  {
    title: "A model reads the shape",
    body: "The wave's shape shifts a little with how hard the heart pushes and how stiff the arteries are. A neural network learned that link from hospital recordings where pressure was measured at the same time.",
  },
];

export default function HomePage() {
  return (
    <>
      <header className="mb-12 max-w-[760px]">
        <h1 className="text-2xl sm:text-3xl">Estimating blood pressure from a pulse of light</h1>
        <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
          Smartwatches and fingertip sensors record a pulse signal called PPG. This student prototype reads five seconds
          of it and estimates systolic and diastolic pressure, with a range that shows how far off it can be. It is not a
          medical device.
        </p>
      </header>

      <RealPulse />

      <section aria-labelledby="how-h" className="mt-24 grid gap-12 md:grid-cols-[minmax(0,1fr)_280px] md:gap-16">
        <div>
          <h2 id="how-h" className="text-xl sm:text-2xl">
            From light to a pressure estimate
          </h2>
          <ol className="mt-8 flex flex-col gap-8">
            {STEPS.map((s, i) => (
              <li key={s.title} className="grid grid-cols-[32px_1fr] gap-4">
                <span className="num pt-0.5 text-sm text-muted-foreground">{i + 1}</span>
                <div className="max-w-[60ch]">
                  <h3 className="text-base">{s.title}</h3>
                  <p className="mt-1 text-muted-foreground">{s.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="self-center">
          <PpgDiagram className="w-full max-w-[280px]" />
        </div>
      </section>

      <section aria-label="Results" className="mt-24">
        <HeadlineResult />
        <div className="mt-8 flex flex-wrap gap-4">
          <Button asChild size="lg">
            <Link to="/predict">Try a sample</Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link to="/compare">Compare models</Link>
          </Button>
        </div>
      </section>

      <section aria-labelledby="limits-h" className="mt-24 border-t border-rule pt-8">
        <h2 id="limits-h" className="text-xl">
          What this can't do
        </h2>
        <div className="measure mt-4 flex flex-col gap-4 text-muted-foreground">
          <p>
            Don't use it to diagnose, monitor or treat any condition, or to change any medication. The models learned from
            intensive-care patients, so they have not been tested on watch or phone signals from healthy people at home.
          </p>
          <p>
            <Link to="/about" className="font-semibold text-ink underline decoration-rule underline-offset-4 hover:decoration-ink">
              Read about the data, the team and the limitations
            </Link>
          </p>
        </div>
      </section>
    </>
  );
}
