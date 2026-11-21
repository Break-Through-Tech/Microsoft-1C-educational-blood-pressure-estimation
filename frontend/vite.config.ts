import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Canonical path (exact drive-letter case). On Windows a shell started in "c:\..." and a config
// resolved as "C:\..." give the same files two module IDs, which bundles React twice and crashes
// every hook ("Cannot read properties of null (reading 'useState')"). Pin root and alias to one spelling.
const ROOT = fs.realpathSync.native(import.meta.dirname);

// Dev: the FastAPI server runs on :8000 and Vite proxies /api to it, so the browser sees one origin.
// Prod: FastAPI serves frontend/dist itself (see backend/main.py), so no proxy is involved.
const API_TARGET = process.env.CUFFLESS_API ?? "http://127.0.0.1:8000";

/**
 * Self-host the MediaPipe Tasks WASM runtime at /mediapipe/wasm/ (served in dev, copied on build),
 * so the camera feature makes no third-party requests. The face model lives in public/mediapipe/.
 */
function mediapipeWasm(): Plugin {
  const src = path.join(ROOT, "node_modules", "@mediapipe", "tasks-vision", "wasm");
  const files = ["vision_wasm_internal.js", "vision_wasm_internal.wasm"];
  const type = (f: string) => (f.endsWith(".wasm") ? "application/wasm" : "text/javascript");
  return {
    name: "mediapipe-wasm",
    configureServer(server) {
      server.middlewares.use("/mediapipe/wasm", (req, res, next) => {
        const f = (req.url ?? "").split("?")[0].replace(/^\//, "");
        if (!files.includes(f)) return next();
        res.setHeader("Content-Type", type(f));
        fs.createReadStream(path.join(src, f)).pipe(res);
      });
    },
    generateBundle() {
      for (const f of files) {
        this.emitFile({ type: "asset", fileName: `mediapipe/wasm/${f}`, source: fs.readFileSync(path.join(src, f)) });
      }
    },
  };
}

export default defineConfig({
  root: ROOT,
  plugins: [react(), tailwindcss(), mediapipeWasm()],
  resolve: { alias: { "@": path.join(ROOT, "src") }, dedupe: ["react", "react-dom"] },
  server: {
    port: 5173,
    proxy: { "/api": { target: API_TARGET, changeOrigin: true } },
  },
  build: { outDir: path.join(ROOT, "dist"), emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
