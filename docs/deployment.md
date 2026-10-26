# Local and online deployment

The React application is a Vite static frontend. AI inference runs in FastAPI on the host that has
the model files and (for GPU inference) a supported NVIDIA GPU. Vercel is suitable for the frontend
only; ordinary Vercel functions and CPU-only web services do not provide the GPU runtime or
persistent model/audio storage required by Seed-VC and Demucs.

## Local Windows setup

Use the existing root `.venv` and keep its installed CUDA PyTorch. If creating a new environment:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-ai.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-kokoro.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The AI requirements do not pin or replace PyTorch. Confirm the printed CUDA check remains `True`
after installing optional Kokoro packages. Install FFmpeg/FFprobe and eSpeak-NG from their official
Windows distributions and add them to `PATH`.

Download models only when you invoke their explicit, confirmation-prompted setup command:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py piper
.\.venv\Scripts\python.exe scripts\download_models.py kokoro --voice af_heart
.\.venv\Scripts\python.exe scripts\download_models.py demucs
powershell -ExecutionPolicy Bypass -File scripts\install_seed_vc.ps1
.\.venv\Scripts\python.exe scripts\download_models.py seed-vc
```

Seed-VC is optional and its setup installs an isolated Python 3.12 dependency environment that
shares the root venv PyTorch instead of downloading a duplicate CUDA wheel. See the respective
[`Kokoro`](adapters/kokoro.md), [`Seed-VC`](adapters/seed-vc.md), and
[`Piper`](adapters/piper.md) guides for licensing and asset details.

Start the backend and frontend in separate terminals from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000
npm --prefix frontend run dev
```

Set the frontend API origin in `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

Run tests and check model readiness:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
Invoke-RestMethod http://localhost:8000/api/models
```

Real model integration tests execute only if their runtime and assets are installed; otherwise they
are skipped. A passing mocked job-pipeline test is not evidence of successful model inference.

## GPU backend deployment

Choose a GPU-capable server/runtime with a supported NVIDIA driver, CUDA-enabled PyTorch/Torchaudio,
Python 3.12, FFmpeg/FFprobe, and enough disk for model snapshots and temporary audio. Build and test
the backend on that exact OS, driver, CUDA, and PyTorch combination; the Windows RTX 4050 result
does not certify a cloud GPU.

Install base backend dependencies, AI dependencies, and Kokoro dependencies in the server's
environment without reinstalling its preconfigured CUDA PyTorch. Create the Seed-VC environment
with `--system-site-packages`, install `backend/requirements-seed-vc.txt`, and point
`SEED_VC_PYTHON` at it. Download each model explicitly using `scripts/download_models.py` after
reviewing the reported sizes and licences.

Run exactly **one Uvicorn worker per GPU**: this project has one in-process job queue with a single
worker thread, not a distributed queue. Multiple web workers would each have separate job state and
could concurrently load separate copies of GPU models. Use the server's process manager to restart
on failure, but do not increase `--workers` without replacing the queue/state architecture.

Configure these environment variables for the deployment:

- `CORS_ORIGINS`: exact HTTPS frontend origin(s), comma-separated.
- `MODELS_DIR`: persistent, private model directory.
- `UPLOAD_DIR` and `OUTPUT_DIR`: persistent, private upload/output directories on a mounted disk.
- `FILE_TTL_HOURS`, `MAX_UPLOAD_MB`, `MAX_AUDIO_MINUTES`, and `MAX_REFERENCE_SECONDS`: storage and
  input limits appropriate to available disk/compute.
- `KOKORO_DEVICE` and `SEED_VC_DEVICE`: `auto`, `cuda`, or `cpu`.
- `SEED_VC_PYTHON`, `SEED_VC_PROJECT_DIR`, `SEED_VC_DIFFUSION_STEPS`, and `SEED_VC_FP16`.
- `VITE_API_URL`: set at Vite build time to the deployed HTTPS backend origin.

Temporary audio is stored under server-generated IDs and is never placed in the frontend build.
Startup cleanup and the periodic TTL cleanup remove expired uploads, outputs, and task metadata.
Use persistent storage mounted at the configured paths; ephemeral storage loses files after restart.
This filesystem/job store is single-instance and does not coordinate multiple replicas.

**The current app does not implement authentication or per-user authorization. Do not expose the
backend publicly until an authentication/access-control layer and abuse/rate limits are added.**
Also terminate HTTPS at a trusted proxy, restrict CORS, cap upload size, and review model licences
for the intended use. Vercel hosts only the frontend; set `VITE_API_URL` in Vercel and allow that
exact frontend origin in backend `CORS_ORIGINS`.
