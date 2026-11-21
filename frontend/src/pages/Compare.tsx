import { useState } from "react";
import type { SplitName } from "@/api";
import { MetricsBarChart, MetricsTable, pivot } from "@/components/Metrics";
import { ChartFrame, ChartSkeleton, EmptyState, ErrorCallout, LinesSkeleton, PageHeader } from "@/components/States";
import { err1 } from "@/lib/format";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/radix";
import { useMetrics } from "@/lib/queries";

export default function ComparePage() {
  const metrics = useMetrics();
  const [split, setSplit] = useState<SplitName>("validation");

  return (
    <>
      <PageHeader title="Compare models">
        How far each model's estimates landed from the measured pressure, next to a baseline that guesses the same number
        for everyone.
      </PageHeader>

      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-4">
        <ToggleGroup type="single" value={split} onValueChange={(v) => v && setSplit(v as SplitName)} aria-label="Data split">
          <ToggleGroupItem value="validation">Validation</ToggleGroupItem>
          <ToggleGroupItem value="test">Test</ToggleGroupItem>
        </ToggleGroup>
        <p className="text-sm text-muted-foreground">
          {split === "validation"
            ? "The primary model is chosen on validation results."
            : "Test results are reported only, never used to choose a model."}
        </p>
      </div>

      {metrics.isPending && (
        <div className="flex flex-col gap-10">
          <LinesSkeleton lines={4} />
          <ChartSkeleton height={320} />
        </div>
      )}
      {metrics.isError && <ErrorCallout title="Couldn't load model results" error={metrics.error} onRetry={() => metrics.refetch()} />}
      {metrics.isSuccess &&
        (() => {
          const models = pivot(metrics.data.rows, split, metrics.data.primary_model);
          if (models.length === 0)
            return (
              <EmptyState title={`No ${split} results yet`}>
                Run the evaluation script to score each model on the {split} recordings, then reload this page.
              </EmptyState>
            );
          const primary = models.find((m) => m.is_primary);
          const base = models.find((m) => m.is_baseline);
          const splitName = split === "validation" ? "validation" : "test";
          const headline =
            primary?.sbp && base?.sbp
              ? `${primary.display_name} misses systolic pressure by ${err1(primary.sbp.mae)} mmHg on average, against ${err1(base.sbp.mae)} for a guess of the average.`
              : "Average error per model.";
          const source = `UCI Cuff-Less Blood Pressure Estimation dataset, ${splitName} split (${(primary?.sbp?.n ?? models[0].sbp?.n ?? 0).toLocaleString()} five-second windows), split by recording.`;
          return (
            <div className="flex flex-col gap-12">
              <ChartFrame
                titleId="chart-h"
                title={headline}
                subtitle={`Error in mmHg on the ${splitName} recordings. MAE is the average size of a miss; RMSE weighs large misses more heavily. Lower is better.`}
                source={source}
                notes="The baseline guesses the training average for every recording; a model has to beat it to be learning anything from the pulse."
              >
                <MetricsBarChart models={models} />
              </ChartFrame>
              <ChartFrame
                titleId="table-h"
                title="The same results as a table."
                subtitle="Sort by any column. The 90% range is the band shown with each estimate."
                source={source}
                notes={
                  split === "test"
                    ? "Range coverage is the share of test windows whose measured value landed inside the range; 90% is the target."
                    : "The 90% range is sized from validation errors, so its coverage is checked on the test split."
                }
              >
                <MetricsTable models={models} split={split} />
              </ChartFrame>
              {models.filter((m) => !m.is_baseline).length === 1 && (
                <p className="measure text-sm text-muted-foreground">
                  Only one trained model is in the app so far. Teammates' models will appear here once they're retrained and
                  scored on the same recordings, so the comparison stays fair.
                </p>
              )}
            </div>
          );
        })()}
    </>
  );
}
