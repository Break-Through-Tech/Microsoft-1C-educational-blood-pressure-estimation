# CufflessAI web app: FastAPI backend + built React frontend in one container (used by Render, render.yaml).
#   docker build -t cufflessai . && docker run -p 8080:8080 cufflessai
# The trained results are not in git: mount them, or set CUFFLESS_RESULTS_URL so start.py downloads them
# (see src/results_bundle.py). Without them the app runs in demo mode.

# --- 1. build the frontend --------------------------------------------------------------------
FROM node:22-bookworm-slim AS web
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. runtime ---------------------------------------------------------------------------------
FROM python:3.12-slim
# PORT: Render sets its own; 8080 is the local default.
# OMP_NUM_THREADS / MALLOC_ARENA_MAX: one inference thread and fewer malloc arenas keep memory well under
# a 512 MB instance; single-window CPU inference doesn't benefit from more threads.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2
# The app dir must be owned by `app` (WORKDIR alone would make it root-owned): at startup the results
# bundle creates results/ and artifacts/ here, which aren't in git.
RUN useradd --create-home --uid 1000 app && install -d -o app -g app /home/app/app
WORKDIR /home/app/app

COPY --chown=app requirements-app.txt ./
RUN pip install -r requirements-app.txt

COPY --chown=app . ./
COPY --chown=app --from=web /build/frontend/dist ./frontend/dist

USER app
EXPOSE 8080
CMD ["python", "scripts/start.py"]
