# CufflessAI web app: FastAPI backend + built React frontend in one container.
# Used by the Hugging Face Space (sdk: docker, app_port: 8080); runs the same anywhere Docker runs:
#   docker build -t cufflessai . && docker run -p 8080:8080 cufflessai

# --- 1. build the frontend --------------------------------------------------------------------
FROM node:22-bookworm-slim AS web
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. runtime ---------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    HOME=/home/user
# Hugging Face runs Spaces as UID 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt ./
RUN pip install -r requirements-app.txt

# App code, plus artifacts/, results/ and data/test_samples.npz if they were uploaded to the Space.
# Without them the app starts in demo mode.
COPY --chown=user . ./
COPY --chown=user --from=web /build/frontend/dist ./frontend/dist

USER user
EXPOSE 8080
CMD ["python", "scripts/start.py", "--host", "0.0.0.0", "--port", "8080"]
