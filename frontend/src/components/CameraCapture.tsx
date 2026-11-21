/**
 * Webcam pulse capture, entirely in the browser. Loaded lazily (its own chunk) so the rest of the app
 * never depends on the camera, MediaPipe or WebAssembly.
 *
 * Privacy: video frames never leave this component. Each frame is reduced to one number (mean green of
 * a forehead patch); only that 1-D series and the measured frame rate are sent, once, to /api/predict.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Camera, CameraOff, Lock, RotateCcw, TriangleAlert } from "lucide-react";
import type { FaceDetector } from "@mediapipe/tasks-vision";
import { MIN_FS } from "@/api";
import { OtherModels } from "@/components/OtherModels";
import { ResultPanel, ResultSkeleton } from "@/components/ResultPanel";
import { ChartFrame, ErrorCallout } from "@/components/States";
import { WaveformChart } from "@/components/WaveformChart";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/misc";
import { usePredict } from "@/lib/queries";
import { foreheadRoi, meanGreen, measuredFps, recentWindow, smoothRoi, toEvenSignal, type Roi, type Sample } from "@/lib/rppg";

const TARGET_SECONDS = 15;
const DETECT_EVERY_MS = 200;
const FACE_LOST_MS = 1000;
const PATCH_W = 32;
const PATCH_H = 16;

type Phase = "intro" | "starting" | "collecting" | "done" | "error";

/**
 * MediaPipe's WebAssembly runtime prints its native logs (glog lines like "W1121 12:00:00 file.cc:12]"
 * and "INFO: Created TensorFlow Lite XNNPACK delegate") through console.error/warn. They are not app
 * errors, so drop exactly those formats once this module loads; every other message passes through.
 */
const NATIVE_LOG = /^(INFO: |[IWE]\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ [\w.]+:\d+\])/;
for (const level of ["error", "warn", "info", "log"] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    if (typeof args[0] === "string" && NATIVE_LOG.test(args[0])) return;
    original(...args);
  };
}

let detectorPromise: Promise<FaceDetector> | null = null;
/** One detector per page load, from self-hosted files (see vite.config.ts and public/mediapipe/). */
function loadDetector(): Promise<FaceDetector> {
  detectorPromise ??= (async () => {
    const { FaceDetector, FilesetResolver } = await import("@mediapipe/tasks-vision");
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    return FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "/mediapipe/blaze_face_short_range.tflite", delegate: "CPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
  })().catch((e) => {
    detectorPromise = null; // allow a retry
    throw e;
  });
  return detectorPromise;
}

function cameraErrorMessage(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Camera access was blocked. Allow the camera for this site (the camera icon in the address bar), then select Start again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found. Connect a webcam, or use a file upload instead.";
  if (name === "NotReadableError") return "The camera is in use by another app. Close it (for example a video call), then try again.";
  return e instanceof Error ? e.message : "The camera couldn't be started.";
}

/** Live sparkline of the raw green signal (as measured, before flipping or cleaning). */
function LiveSignal({ samples }: { samples: Sample[] }) {
  const pts = recentWindow(samples, 8);
  if (pts.length < 2) return <div className="flex h-24 items-center text-sm text-muted-foreground">Waiting for a face…</div>;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of pts) {
    lo = Math.min(lo, p.g);
    hi = Math.max(hi, p.g);
  }
  const span = Math.max(hi - lo, 0.05);
  const t0 = pts[0].t;
  const d = pts.map((p, i) => `${i ? "L" : "M"}${(((p.t - t0) / 8) * 300).toFixed(1)},${(90 - ((p.g - lo) / span) * 80).toFixed(1)}`).join("");
  return (
    <svg viewBox="0 0 300 100" preserveAspectRatio="none" className="h-24 w-full" role="img" aria-label="Live forehead green signal">
      <path d={d} fill="none" className="stroke-truth" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function ProgressRing({ value, label }: { value: number; label: string }) {
  const r = 30;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative size-20 shrink-0" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)} aria-label="Collection progress">
      <svg viewBox="0 0 72 72" className="size-20 -rotate-90">
        <circle cx={36} cy={36} r={r} fill="none" className="stroke-rule" strokeWidth={5} />
        <circle
          cx={36}
          cy={36}
          r={r}
          fill="none"
          className="stroke-led transition-[stroke-dashoffset] duration-200"
          strokeWidth={5}
          strokeDasharray={c}
          strokeDashoffset={c * (1 - value)}
        />
      </svg>
      <span className="num absolute inset-0 flex items-center justify-center text-sm font-semibold">{label}</span>
    </div>
  );
}

export default function CameraCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const patchRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const samplesRef = useRef<Sample[]>([]);
  const roiRef = useRef<Roi | null>(null);
  const runningRef = useRef(false);
  const lastDetectRef = useRef(0);
  const lastFaceRef = useRef(0);
  const lastUiRef = useRef(0);

  const [phase, setPhase] = useState<Phase>("intro");
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<{ samples: Sample[]; face: boolean; dark: boolean; restarted: boolean }>({ samples: [], face: false, dark: false, restarted: false });
  const [sent, setSent] = useState<{ signal: number[]; fs: number } | null>(null);
  const predict = usePredict();

  const supported = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && window.isSecureContext;

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]); // always release the camera on unmount

  const finish = useCallback(() => {
    stopCamera();
    const samples = samplesRef.current;
    const fs = measuredFps(samples);
    if (fs < MIN_FS) {
      setError(
        `The camera delivered only ${fs.toFixed(1)} frames per second; at least ${MIN_FS} are needed. Add light (cameras slow down in the dark) or close other apps using the camera, then try again.`,
      );
      setPhase("error");
      return;
    }
    const signal = toEvenSignal(samples, fs);
    const rounded = Math.round(fs * 1000) / 1000;
    setSent({ signal, fs: rounded });
    setPhase("done");
    predict.mutate({ signal, fs: rounded });
  }, [predict, stopCamera]);

  const onFrame = useCallback(
    (detector: FaceDetector) => {
      const video = videoRef.current;
      if (!runningRef.current || !video || video.readyState < 2) return;
      const now = performance.now();
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      if (now - lastDetectRef.current > DETECT_EVERY_MS) {
        lastDetectRef.current = now;
        const det = detector.detectForVideo(video, now).detections[0];
        if (det?.boundingBox) {
          const roi = foreheadRoi({ box: det.boundingBox, keypoints: det.keypoints ?? [] }, vw, vh);
          if (roi) {
            roiRef.current = smoothRoi(roiRef.current, roi);
            lastFaceRef.current = now;
          }
        }
      }

      const faceVisible = now - lastFaceRef.current < FACE_LOST_MS;
      let restarted = false;
      if (!faceVisible && samplesRef.current.length > 0) {
        // a gap would put a jump in the signal: start the 15 s over
        samplesRef.current = [];
        restarted = true;
      }

      const roi = roiRef.current;
      let dark = false;
      if (faceVisible && roi) {
        patchRef.current ??= Object.assign(document.createElement("canvas"), { width: PATCH_W, height: PATCH_H });
        const ctx = patchRef.current.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(video, roi.x, roi.y, roi.w, roi.h, 0, 0, PATCH_W, PATCH_H);
        const g = meanGreen(ctx.getImageData(0, 0, PATCH_W, PATCH_H).data);
        dark = g < 30;
        samplesRef.current.push({ t: now / 1000, g });
      }

      // overlay: the forehead patch being measured
      const ov = overlayRef.current;
      if (ov) {
        if (ov.width !== vw) ov.width = vw;
        if (ov.height !== vh) ov.height = vh;
        const octx = ov.getContext("2d")!;
        octx.clearRect(0, 0, vw, vh);
        if (roi && faceVisible) {
          octx.strokeStyle = "#0b7a53";
          octx.lineWidth = Math.max(2, vw / 240);
          octx.strokeRect(roi.x, roi.y, roi.w, roi.h);
        }
      }

      const s = samplesRef.current;
      const elapsed = s.length > 1 ? s[s.length - 1].t - s[0].t : 0;
      if (now - lastUiRef.current > 100 || restarted) {
        lastUiRef.current = now;
        setLive({ samples: s.slice(-400), face: faceVisible, dark, restarted });
      }
      if (elapsed >= TARGET_SECONDS) finish();
    },
    [finish],
  );

  const start = useCallback(async () => {
    setError(null);
    setSent(null);
    predict.reset();
    samplesRef.current = [];
    roiRef.current = null;
    lastFaceRef.current = 0;
    setLive({ samples: [], face: false, dark: false, restarted: false });
    setPhase("starting");
    try {
      const [detector, stream] = await Promise.all([
        loadDetector().catch((e) => {
          throw new Error(`The face detector couldn't load (${e instanceof Error ? e.message : "unknown error"}). File upload still works.`);
        }),
        navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
          audio: false,
        }),
      ]);
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      runningRef.current = true;
      setPhase("collecting");

      const hasRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
      const loop = () => {
        if (!runningRef.current) return;
        onFrame(detector);
        if (!runningRef.current) return;
        if (hasRvfc) video.requestVideoFrameCallback(loop);
        else requestAnimationFrame(loop);
      };
      loop();
    } catch (e) {
      stopCamera();
      setError(e instanceof Error && !(e instanceof DOMException) ? e.message : cameraErrorMessage(e));
      setPhase("error");
    }
  }, [onFrame, predict, stopCamera]);

  const cancel = () => {
    stopCamera();
    setPhase("intro");
  };

  const elapsed = useMemo(() => {
    const s = live.samples;
    return s.length > 1 ? s[s.length - 1].t - s[0].t : 0;
  }, [live.samples]);

  if (!supported)
    return (
      <Alert>
        <CameraOff />
        <div>
          <p className="font-semibold">The camera isn't available here</p>
          <p className="text-muted-foreground">
            This browser doesn't allow camera access on this page (it needs a secure connection, such as https or
            localhost). File upload still works.
          </p>
        </div>
      </Alert>
    );

  const showCamera = phase === "starting" || phase === "collecting";

  return (
    <div className="flex flex-col gap-8">
      {(phase === "intro" || phase === "error") && (
        <div className="frame flex flex-col gap-6 p-6">
          <div className="measure flex flex-col gap-4">
            <h2 className="text-lg">Measure your pulse with the camera</h2>
            <p className="text-muted-foreground">
              Each heartbeat slightly changes the colour of your skin, because blood absorbs green light. With good light,
              a webcam can pick this up from your forehead.
            </p>
            <div className="flex gap-2 text-sm">
              <Lock className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
              <p>
                <span className="font-semibold">Why the camera permission:</span> your browser will ask before the camera
                turns on. The video stays on this device. Each frame is reduced to one number, the average green
                brightness of your forehead, and then discarded. Only that list of numbers and the frame rate are sent,
                once, to make the estimate. No video or image is uploaded, and the server doesn't store the signal. The
                camera turns off as soon as the 15 seconds are up.
              </p>
            </div>
            <p className="text-sm text-muted-foreground">
              For the best signal: face a window or lamp, keep your forehead uncovered, and stay still for 15 seconds.
            </p>
          </div>
          {phase === "error" && error && <ErrorCallout title="Camera capture stopped" error={new Error(error)} />}
          <div>
            <Button size="lg" onClick={start}>
              <Camera /> {phase === "error" ? "Try again" : "Start the camera"}
            </Button>
          </div>
        </div>
      )}

      {/* the video element is always mounted so the ref exists when the stream arrives */}
      <div className={showCamera ? "grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" : "hidden"}>
        <div className="frame relative overflow-hidden bg-black">
          <video ref={videoRef} muted playsInline className="block aspect-[4/3] w-full -scale-x-100 object-cover" aria-label="Your camera preview (stays on this device)" />
          <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover" aria-hidden />
          {phase === "starting" && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm text-white">Starting the camera and face detector…</div>
          )}
        </div>
        <div className="frame flex flex-col gap-4 p-6">
          <div className="flex items-center gap-4">
            <ProgressRing value={Math.min(1, elapsed / TARGET_SECONDS)} label={`${Math.max(0, Math.ceil(TARGET_SECONDS - elapsed))} s`} />
            <div className="text-sm">
              <p className="font-semibold">
                {!live.face ? "Looking for your face" : live.dark ? "It's too dark" : "Hold still"}
              </p>
              <p className="text-muted-foreground">
                {!live.face
                  ? "Centre your face in the frame. The green box shows the patch being measured."
                  : live.dark
                    ? "Turn on a light or face a window; the signal is too faint."
                    : live.restarted
                      ? "Your face left the frame, so the count started over."
                      : `Collecting ${TARGET_SECONDS} seconds of pulse from your forehead.`}
              </p>
            </div>
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Raw signal: average forehead green, last 8 seconds</p>
            <LiveSignal samples={live.samples} />
          </div>
          <div>
            <Button variant="outline" onClick={cancel}>
              <CameraOff /> Stop the camera
            </Button>
          </div>
        </div>
      </div>

      {phase === "done" && (
        <div className="flex flex-col gap-8" aria-live="polite">
          <Alert variant="caution" className="border border-caution/40 py-4">
            <TriangleAlert />
            <div className="flex flex-col gap-1">
              <p className="text-base font-semibold">These numbers are not reliable</p>
              <p className="text-muted-foreground">
                The models learned from fingertip sensors on hospital patients. A camera watching a forehead records a
                much weaker, noisier version of the pulse, affected by light, skin tone and movement, so the signal is
                unlike anything the models were trained on. Treat the estimates below as a demonstration only.
              </p>
            </div>
          </Alert>

          {sent && (
            <ChartFrame
              title={`What was sent: ${sent.signal.length} numbers at ${sent.fs.toFixed(1)} frames per second.`}
              subtitle="Average forehead green per frame (0–255 scale, shown centred on zero), evened out in time and flipped so peaks mean more blood, as in the training data. The server resamples it to 125 Hz and cleans it like any other signal."
              source="Your camera, processed in this browser. No video was uploaded; the server doesn't store the signal."
            >
              <WaveformChart
                label="Camera pulse signal that was sent"
                yLabel="Green, centred"
                height={200}
                points={(() => {
                  // centred for display only; the values sent are unchanged
                  const mean = sent.signal.reduce((a, b) => a + b, 0) / sent.signal.length;
                  return sent.signal.map((v, i) => ({ t: i / sent.fs, v: v - mean }));
                })()}
              />
            </ChartFrame>
          )}

          {predict.isPending && <ResultSkeleton />}
          {predict.isError && (
            <ErrorCallout title="Couldn't estimate from the camera signal" error={predict.error} onRetry={sent ? () => predict.mutate(sent) : undefined} />
          )}
          {predict.isSuccess && (
            <>
              <div className="frame flex flex-wrap items-baseline gap-x-4 gap-y-2 px-6 py-6">
                <p className="text-sm font-semibold">Heart rate from the camera</p>
                <p className="num text-[2.5rem] leading-none font-semibold">
                  {predict.data.quality.heart_rate_bpm != null ? Math.round(predict.data.quality.heart_rate_bpm) : "n/a"}
                  <span className="ml-2 text-sm font-normal text-muted-foreground">beats per minute</span>
                </p>
                <p className="w-full text-sm text-muted-foreground">
                  Counted from the peaks in the cleaned signal. If it's far from your actual pulse, the camera mostly
                  picked up noise, and the estimates below are even less meaningful.
                </p>
              </div>
              <ResultPanel result={predict.data} />
              <OtherModels result={predict.data} />
            </>
          )}
          <div>
            <Button variant="outline" onClick={start}>
              <RotateCcw /> Measure again
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
