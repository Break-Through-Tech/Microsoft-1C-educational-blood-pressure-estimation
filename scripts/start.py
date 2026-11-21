"""Production start: build the frontend if needed, then serve the API and frontend/dist with uvicorn.

    python scripts/start.py            # host 0.0.0.0, port $PORT or 8080
    python scripts/start.py --rebuild  # force a fresh frontend build first

Real mode if results/, artifacts/ and data/test_samples.npz are present and loadable; otherwise the
app starts in demo mode (it never crashes on missing or broken files). If CUFFLESS_RESULTS_URL is set
and those files are missing, they are downloaded first (see src/results_bundle.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"


def build_frontend(force: bool) -> None:
    if DIST_INDEX.is_file() and not force:
        print(f"[start] frontend already built ({DIST_INDEX.relative_to(ROOT)})", flush=True)
        return
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("[start] frontend/dist is missing and npm was not found. Install Node.js 20.19+ "
                 "(or 22.12+), or build the frontend elsewhere and copy frontend/dist here.")
    if not (FRONTEND / "node_modules").is_dir():
        print("[start] installing frontend dependencies (npm ci)...", flush=True)
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=FRONTEND, check=True)
    print("[start] building the frontend (npm run build)...", flush=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)
    if not DIST_INDEX.is_file():
        sys.exit("[start] the build finished but frontend/dist/index.html is missing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the frontend even if dist exists")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = ap.parse_args()

    try:
        build_frontend(args.rebuild)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[start] frontend build failed (exit {e.returncode}); see the output above")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # hosts that build only from git (Render) fetch the trained results at startup, if configured
    from src.results_bundle import fetch_if_configured
    fetch_if_configured()
    import uvicorn

    print(f"[start] serving on http://{args.host}:{args.port}", flush=True)
    # one worker: the model and samples are loaded once per process at startup
    uvicorn.run("backend.main:app", host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
