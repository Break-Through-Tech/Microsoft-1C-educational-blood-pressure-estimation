import { Component, Suspense, lazy, useId, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import { Activity, FileUp, Lock, TriangleAlert, Video } from "lucide-react";
import { MAX_FS, MAX_SECONDS, MIN_FS, MIN_SECONDS } from "@/api";
import { OtherModels } from "@/components/OtherModels";
import { ResultPanel, ResultSkeleton } from "@/components/ResultPanel";
import { ChartFrame, ChartSkeleton, EmptyState, ErrorCallout, PageHeader } from "@/components/States";
import { WaveformChart } from "@/components/WaveformChart";
import { Button } from "@/components/ui/button";
import { Alert, Input, Label } from "@/components/ui/misc";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/radix";
import { decimateForPlot, parseCsv, signalProblem, type ParsedCsv } from "@/lib/csv";
import { useModels, usePredict } from "@/lib/queries";
import { cn } from "@/lib/utils";

const MAX_FILE_BYTES = 20 * 1024 * 1024;

// The camera feature is its own chunk (MediaPipe + WebAssembly); nothing else depends on it.
const CameraCapture = lazy(() => import("@/components/CameraCapture"));

/** If the camera code fails to load or crashes, contain it to this tab. */
class CameraBoundary extends Component<{ children: ReactNode }, { failed: Error | null }> {
  state = { failed: null as Error | null };
  static getDerivedStateFromError(failed: Error) {
    return { failed };
  }
  render() {
    if (this.state.failed)
      return (
        <ErrorCallout
          title="The camera feature couldn't start"
          error={new Error(`${this.state.failed.message}. File upload still works; switch to the File tab.`)}
          onRetry={() => this.setState({ failed: null })}
        />
      );
    return this.props.children;
  }
}

function CsvDropzone({ onFile, fileName }: { onFile: (f: File) => void; fileName: string | null }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const id = useId();
  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={cn(
        "flex flex-col items-start gap-4 rounded-md border border-dashed px-6 py-8 transition-colors sm:flex-row sm:items-center",
        over ? "border-led bg-led-soft" : "border-baseline bg-surface",
      )}
    >
      <FileUp className="size-8 shrink-0 text-muted-foreground" aria-hidden />
      <div className="flex-1">
        <p className="font-semibold">{fileName ?? "Drop a CSV file here"}</p>
        <p className="text-sm text-muted-foreground">
          One column of PPG values, with or without a header. A time column is optional and is used to work out the
          sampling rate.
        </p>
      </div>
      <input
        ref={input}
        id={id}
        type="file"
        accept=".csv,.txt,.tsv,text/csv,text/plain"
        className="sr-only"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      <Button variant="outline" onClick={() => input.current?.click()}>
        {fileName ? "Choose another file" : "Choose file"}
      </Button>
    </div>
  );
}

export default function TryYourOwnPage() {
  const [params, setParams] = useSearchParams();
  const models = useModels();
  const predict = usePredict();
  const [fileName, setFileName] = useState<string | null>(null);
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [column, setColumn] = useState(0);
  const [fsText, setFsText] = useState("125");

  const onFile = async (f: File) => {
    predict.reset();
    setFileName(f.name);
    setParsed(null);
    setParseError(null);
    if (f.size > MAX_FILE_BYTES) {
      setParseError(`This file is ${(f.size / 1e6).toFixed(0)} MB. Use a file under 20 MB (${MAX_SECONDS} s of signal is far smaller).`);
      return;
    }
    try {
      const p = parseCsv(await f.text());
      setParsed(p);
      setColumn(0);
      if (p.inferredFs) setFsText(String(p.inferredFs));
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "This file couldn't be read as CSV.");
    }
  };

  const fs = Number(fsText);
  const signal = parsed?.columns[column]?.values ?? null;
  const problem = signal ? signalProblem(signal, fs) : null;
  const plot = useMemo(() => (signal && Number.isFinite(fs) && fs > 0 ? decimateForPlot(signal, fs) : []), [signal, fs]);
  const col = parsed?.columns[column];

  return (
    <>
      <PageHeader title="Try your own signal">
        Upload a PPG recording from a pulse sensor, or measure your pulse with your webcam. Either way the signal is
        checked in your browser first, then sent once to make the estimate, and never stored.
      </PageHeader>

      <Tabs
        value={params.get("tab") === "camera" ? "camera" : "file"}
        onValueChange={(v) => setParams(v === "camera" ? { tab: "camera" } : {}, { replace: true })}
      >
        <TabsList>
          <TabsTrigger value="file">
            <FileUp className="size-4" aria-hidden /> File
          </TabsTrigger>
          <TabsTrigger value="camera">
            <Video className="size-4" aria-hidden /> Camera
          </TabsTrigger>
        </TabsList>
        <TabsContent value="file">
          <div className="flex flex-col gap-8">
            <CsvDropzone onFile={onFile} fileName={fileName} />

            {parseError && (
              <Alert variant="caution">
                <TriangleAlert />
                <div>
                  <p className="font-semibold">This file can't be used</p>
                  <p className="text-muted-foreground">{parseError}</p>
                </div>
              </Alert>
            )}

            {parsed && (
              <div className="flex flex-col gap-6 sm:flex-row sm:items-end">
                {parsed.columns.length > 1 && (
                  <div className="flex flex-col gap-2">
                    <Label htmlFor="col-select">Signal column</Label>
                    <Select value={String(column)} onValueChange={(v) => { setColumn(Number(v)); predict.reset(); }}>
                      <SelectTrigger id="col-select" className="w-full sm:w-56">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {parsed.columns.map((c, i) => (
                          <SelectItem key={c.name + i} value={String(i)}>
                            {c.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <div className="flex flex-col gap-2">
                  <Label htmlFor="fs-input">Sampling rate (Hz)</Label>
                  <Input
                    id="fs-input"
                    inputMode="decimal"
                    className="w-full sm:w-40"
                    value={fsText}
                    aria-invalid={!Number.isFinite(fs) || fs < MIN_FS || fs > MAX_FS}
                    aria-describedby="fs-help"
                    onChange={(e) => {
                      setFsText(e.target.value);
                      predict.reset();
                    }}
                  />
                </div>
                <p id="fs-help" className="text-xs text-muted-foreground sm:pb-4">
                  {parsed.inferredFs
                    ? `Worked out from the "${parsed.timeColumn}" column. Change it if that's wrong.`
                    : `Between ${MIN_FS} and ${MAX_FS} Hz. The models were trained at 125 Hz; other rates are resampled.`}
                </p>
              </div>
            )}

            {signal && col && plot.length > 1 && (
              <ChartFrame
                title={`Your signal: ${signal.length.toLocaleString()} values${Number.isFinite(fs) && fs > 0 ? `, ${(signal.length / fs).toFixed(1)} seconds` : ""}.`}
                subtitle={`Column "${col.name}" as uploaded, before cleaning.${col.invalid > 0 ? ` Skipped ${col.invalid} empty or non-numeric rows.` : ""}`}
                source={`${fileName}, read in your browser. Nothing is stored.`}
              >
                <WaveformChart label={`Uploaded signal from ${fileName}`} yLabel="As uploaded" points={plot} height={220} />
              </ChartFrame>
            )}

            {signal && problem && (
              <Alert variant="caution">
                <TriangleAlert />
                <p>{problem}</p>
              </Alert>
            )}

            {signal && (
              <div className="flex flex-col gap-4">
                <div>
                  <Button size="lg" disabled={!!problem || predict.isPending} onClick={() => predict.mutate({ signal, fs })}>
                    <Activity /> {predict.isPending ? "Estimating…" : "Estimate blood pressure"}
                  </Button>
                </div>
                <p className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Lock className="size-3.5" aria-hidden /> Your file is sent once to make the estimate and is never saved.
                </p>
              </div>
            )}

            {!parsed && !parseError && (
              <EmptyState title="No file yet">
                Signals need to be {MIN_SECONDS} to {MAX_SECONDS} seconds long. Longer recordings work better, because the
                filter is applied to the whole signal before the middle {MIN_SECONDS} seconds are used.
              </EmptyState>
            )}
          </div>

          <div className="mt-10" aria-live="polite">
            {predict.isPending && <ResultSkeleton />}
            {predict.isError && (
              <ErrorCallout
                title="Couldn't estimate blood pressure"
                error={predict.error}
                onRetry={signal && !problem ? () => predict.mutate({ signal, fs }) : undefined}
              />
            )}
            {predict.isSuccess && (
              <>
                <ResultPanel result={predict.data} selectionRule={models.data?.selection_rule} />
                <OtherModels result={predict.data} />
              </>
            )}
          </div>
        </TabsContent>
        <TabsContent value="camera">
          <CameraBoundary>
            <Suspense fallback={<ChartSkeleton height={240} />}>
              <CameraCapture />
            </Suspense>
          </CameraBoundary>
        </TabsContent>
      </Tabs>
    </>
  );
}
