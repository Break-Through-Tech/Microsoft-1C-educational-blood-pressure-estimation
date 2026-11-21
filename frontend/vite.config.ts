import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Canonical path (exact drive-letter case). On Windows a shell started in "c:\..." and a config
// resolved as "C:\..." give the same files two module IDs, which bundles React twice and crashes
// every hook ("Cannot read properties of null (reading 'useState')"). Pin root and alias to one spelling.
const ROOT = fs.realpathSync.native(import.meta.dirname);

// Dev: the FastAPI server runs on :8000 and Vite proxies /api to it, so the browser sees one origin.
// Prod: FastAPI serves frontend/dist itself (see backend/main.py), so no proxy is involved.
const API_TARGET = process.env.CUFFLESS_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  root: ROOT,
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.join(ROOT, "src") }, dedupe: ["react", "react-dom"] },
  server: {
    port: 5173,
    proxy: { "/api": { target: API_TARGET, changeOrigin: true } },
  },
  build: { outDir: path.join(ROOT, "dist"), emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
