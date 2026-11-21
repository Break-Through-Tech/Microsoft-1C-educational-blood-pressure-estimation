import type { ReactNode } from "react";
import { Link } from "react-router";
import { ErrorCallout, LinesSkeleton, PageHeader } from "@/components/States";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/misc";
import { useHealth, useModels } from "@/lib/queries";
import { TEAM } from "@/lib/team";

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="grid grid-cols-[minmax(0,1fr)] gap-4 border-t border-rule py-12 md:grid-cols-[220px_minmax(0,1fr)] md:gap-12">
      <h2 id={id} className="text-xl">
        {title}
      </h2>
      <div className="measure flex min-w-0 flex-col gap-4">{children}</div>
    </section>
  );
}

function ModelsInApp() {
  const models = useModels();
  const health = useHealth();
  if (models.isPending) return <LinesSkeleton lines={2} />;
  if (models.isError) return <ErrorCallout title="Couldn't load the model list" error={models.error} onRetry={() => models.refetch()} />;
  return (
    <>
      <ul className="flex flex-col gap-2">
        {models.data.models.map((m) => (
          <li key={m.name} className="flex flex-wrap items-center gap-2">
            <span className={m.is_baseline ? "text-muted-foreground italic" : "font-medium"}>{m.display_name}</span>
            {m.is_primary && <Badge variant="led">Primary</Badge>}
            {m.is_demo && <Badge variant="caution">Demo placeholder</Badge>}
          </li>
        ))}
      </ul>
      {health.data?.notes.map((n) => (
        <p key={n} className="text-sm text-muted-foreground">
          {n[0].toUpperCase() + n.slice(1)}.
        </p>
      ))}
    </>
  );
}

function BandLimitation() {
  const models = useModels();
  const primary = models.data?.models.find((m) => m.is_primary);
  const b = primary?.bands;
  return (
    <li>
      Errors are large.{" "}
      {b
        ? `The primary model's 90% range is about ±${Math.round(b.sbp.half_width_mmHg)} mmHg systolic and ±${Math.round(b.dbp.half_width_mmHg)} mmHg diastolic, far too wide for any medical decision.`
        : "The likely range around each estimate is far too wide for any medical decision."}
    </li>
  );
}

export default function AboutPage() {
  return (
    <>
      <PageHeader title="About this project">
        CufflessAI was built by a student team in Break Through Tech AI Studio, with Microsoft as the host company. It's
        judged on being a complete, honest and reproducible prototype, not on clinical accuracy.
      </PageHeader>

      <Section id="data-h" title="The data">
        <p>
          The models learn from the{" "}
          <a
            href="https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation"
            className="font-medium text-led underline-offset-4 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            UCI Cuff-Less Blood Pressure Estimation dataset
          </a>
          : about 12,000 hospital recordings from intensive-care patients, with a finger PPG and an arterial line
          measuring pressure at the same time, both at 125 Hz.
        </p>
        <p>
          Each recording is cleaned as a whole (mean removed, then a 0.5 to 5 Hz filter), cut into 5-second windows, and
          each window is scaled to the range −1 to 1. The reference pressures come from the arterial line in the same
          window.
        </p>
        <p>
          Recordings are split 70 / 15 / 15 into training, validation and test, so no recording appears in more than one
          split. The dataset doesn't say which recordings come from the same person, so one person could appear in more
          than one split. The results are not proven to hold for new people.
        </p>
      </Section>

      <Section id="models-h" title="Models in the app">
        <ModelsInApp />
        <p className="text-sm text-muted-foreground">
          The primary model is the one with the lowest validation error. Test results are reported on the{" "}
          <Link to="/compare" className="font-medium text-led underline-offset-4 hover:underline">
            Compare page
          </Link>{" "}
          but never used to choose.
        </p>
      </Section>

      <Section id="team-h" title="The team">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-0">Name</TableHead>
              <TableHead>Contribution</TableHead>
              <TableHead>Models built</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {TEAM.map((m) => (
              <TableRow key={m.name}>
                <TableCell className="pl-0 font-medium whitespace-nowrap">{m.name}</TableCell>
                <TableCell className="min-w-56 text-muted-foreground">{m.role}</TableCell>
                <TableCell className="min-w-48">
                  {m.models ? (
                    <ul className="flex flex-col gap-1">
                      {m.models.map((x) => (
                        <li key={x.name}>
                          <span className="font-medium">{x.name}</span>
                          <span className="block text-xs text-muted-foreground">
                            {x.detail}. {x.inApp ? "In the app." : "Not in the app yet."}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-muted-foreground">None</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <p className="text-sm text-muted-foreground">
          Models trained on a different split are added only after they're retrained on this app's split, so every
          model is scored on recordings none of them have seen.
        </p>
      </Section>

      <Section id="limits-h" title="Limitations">
        <ul className="flex list-disc flex-col gap-4 pl-6">
          <BandLimitation />
          <li>
            The training data comes from intensive-care patients with an arterial line. Signals from phones, watches or
            healthy people at home look different, and the model hasn't been tested on them.
          </li>
          <li>The split is by recording, not by person, so results may be optimistic for someone new.</li>
          <li>
            Estimates drift toward the average: very high pressures tend to be underestimated and very low ones
            overestimated.
          </li>
          <li>
            Explanations show where the model is sensitive, not why. They don't prove a physiological cause.
          </li>
          <li>
            The app shows numbers only. It doesn't label any reading as normal or high, because it can't support that
            judgment.
          </li>
        </ul>
      </Section>

      <Section id="privacy-h" title="Privacy">
        <p>
          A file you upload on Try your own is read in your browser. Only the signal values are sent to the server, once,
          to make the estimate. They're processed in memory and never saved or logged. No camera video is collected.
        </p>
        <p>The app has no accounts, cookies or analytics. Your theme choice is kept in your browser.</p>
      </Section>
    </>
  );
}
