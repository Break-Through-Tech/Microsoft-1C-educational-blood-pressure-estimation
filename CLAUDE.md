# CLAUDE.md

## Project
CufflessAI: an educational prototype that estimates systolic/diastolic blood pressure from PPG/ECG
signals (Break Through Tech AI Studio × Microsoft). Not a medical device. It is judged on being a
complete, honest, reproducible end-to-end prototype, not on clinical accuracy.

## Data and model conventions
- **Standard input** (taken from Claire's ResNet v3); every model added later must accept it:
  - PPG only, 1 channel, `TARGET_FS = 125` Hz (UCI native rate, no resampling)
  - window = 625 samples (5 s), non-overlapping; tensor shape `(n, 1, 625)` float32
  - preprocessing, in this order:
    1. mean removal on the **whole recording**
    2. Butterworth band-pass, order 4, 0.5–5 Hz, zero-phase, also on the whole recording
    3. cut the 5 s windows
    4. divide each window by its max absolute value, so it lies in [-1, 1]
  - no dataset-wide input stats: filter the longest available signal, never a lone 5 s clip
    (edge effects)
- Units are mmHg everywhere; predictions are always `[SBP, DBP]`, shape `(n, 2)`.
- Split by **recording**, 70/15/15, saved to `data/splits.json`. UCI has no patient IDs, so recording
  is the closest proxy to subject. Never describe the split as patient-independent.
- A model trained on a different split (e.g. Claire's checkpoints) may only be scored on recordings
  that were held out from *its own* training. Retrain it on `data/splits.json` or report it separately.
- Metrics: MAE and RMSE for SBP and DBP, always alongside a mean baseline (predict the training mean).
- Primary model = lowest validation MAE_SBP + MAE_DBP, recorded in `results/best_model.json`.
  Test metrics are for reporting only, never for choosing.

## Layout
- `notebooks/`: teammates' originals copied via `git show`. Read-only, local only; never edit.
- `src/`: pipeline (`preprocess.py`, `split.py`), model wrappers (`src/models/`), `registry.py`
- `scripts/`: training and evaluation
- `backend/`: FastAPI; the API contract is `backend/schemas.py`
- `frontend/`: React + Vite + TypeScript + Tailwind + shadcn/ui + Recharts; types mirror `schemas.py`
- Local only (in `.git/info/exclude`): `notebooks/`, `artifacts/`, `results/`, `data/`

## Model interface
- `BPModel`: `name`, `input_kind` (`"waveform"` | `"features"`), `predict(X) -> (n, 2)` mmHg,
  `explain(X) -> dict`, `load(path)`
- PyTorch models run on CPU in the app, `model.eval()` under `torch.inference_mode()`.
- Claire's `.pt` files are dicts (`model` state_dict, `y_mean`, `y_std`, `config`, `history`).
  - Output is standardized: mmHg = `out * y_std + y_mean`.
  - Loading needs `weights_only=False` (numpy arrays are pickled), so load only team-trusted files.

## Modes
- **Real mode** when `results/best_model.json` and its artifacts exist; otherwise **demo mode**, with
  placeholder models and synthetic data.
- Every API response includes `"mode"`; the frontend shows a "Demo data" badge in demo mode.

## App rules
- Always show the "educational only, not a medical device" disclaimer.
- Error bands = 90th percentile of |residual| on the VALIDATION split, per model and per target
  (split-conformal style). Test residuals are only used to report band coverage (the % of test
  samples whose true value falls inside the band), never to size the bands.
- Temporary fallback: if a model has no validation residuals yet (e.g. the current ResNet, which has
  only test residuals), use its test residuals, set `"band_source": "test (provisional)"` in the API
  response, and show a small "provisional" note in the UI. Remove the fallback once `evaluate.py`
  produces validation residuals for every model.
- Never store user-submitted signals or video.
- No diagnostic labels (e.g. "hypertensive", AHA stages) on predictions.

## Git
- Work only on `app-demo`. Bring in teammates' files with `git show origin/<branch>:<path>`; never
  merge, rebase or check out their branches.
- Never commit anything under the local-only folders.

## Commands
- Install (Python 3.12): `py -3.12 -m venv .venv`, then
  `.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
- Data: unzip `Part_1.mat`–`Part_4.mat` from https://archive.ics.uci.edu/dataset/340 into
  `data/cuff_less_bp_estimation/`. The first `load_dataset()` builds `data/cache/` (several minutes).
- Split: `.venv/Scripts/python -m src.split` (refuses to save unless it reproduces the ResNet's split)
- Train / evaluate: `.venv/Scripts/python scripts/evaluate.py`
- Run backend: `.venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000` (docs at `/docs`;
  CORS origins via `CUFFLESS_CORS_ORIGINS`, default the Vite dev server on port 5173)
- Run frontend (dev, Node 20+): `cd frontend && npm install && npm run dev` (Vite on 5173 proxies `/api` to
  the backend on 8000; override with `CUFFLESS_API`)
- Build: `cd frontend && npm run build` → `frontend/dist`; then the backend alone serves the app at
  `http://localhost:8000` (`.venv/Scripts/python -m uvicorn backend.main:app --port 8000`), with unknown non-API
  paths falling back to `index.html`
- Test: `.venv/Scripts/python -m pytest` (only `tests/`; data-dependent tests skip when files are missing)
