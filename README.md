# Vocal Studio

An independent AI audio platform inspired by the kinds of tools Kits AI offers.
Not affiliated with or endorsed by Kits AI. No proprietary models or code are used.

## Status: Vocal Studio AI (Demucs, Piper, Kokoro, Seed-VC, and RVC)
Working: health check, validated audio upload, private playback, model status, dashboard UI,
background jobs, vocal separation (Demucs), text-to-speech (Piper and Kokoro), **reference-conditioned voice
conversion (Seed-VC: source audio + target speaker sample)**, and legacy RVC checkpoint conversion
(`POST /api/audio/convert-rvc`). Seed-VC and RVC each use an isolated Python environment so Demucs/
Piper PyTorch dependencies stay intact. Models are reported available only when their runtime and
required local assets are present. Voice Lab and projects/history are not built yet.

## Enable the AI features (Windows, backend venv active, PyTorch with CUDA already installed)
    cd backend
    pip install -r requirements-ai.txt
    cd ..
    python scripts/check_env.py
    python scripts/download_models.py demucs     # asks before downloading
    python scripts/download_models.py piper      # asks before downloading
Then restart the backend. Licence notes: docs/adapters/. Test details: docs/milestone-2-3.md.

### Enable Kokoro text-to-speech
In the existing backend environment, install [backend/requirements-kokoro.txt](backend/requirements-kokoro.txt)
without changing the CUDA PyTorch installation, then follow
[docs/adapters/kokoro.md](docs/adapters/kokoro.md). Download model assets explicitly with
`python scripts/download_models.py kokoro --voice af_heart`; other supported English voices can be
requested by repeating `--voice`. Piper remains available as an independent CPU alternative.

### Enable Seed-VC reference voice conversion (primary Voice Conversion page)
Seed-VC converts source speech toward a short target-speaker reference recording (zero-shot). Follow
[docs/adapters/seed-vc.md](docs/adapters/seed-vc.md) or run `scripts/install_seed_vc.ps1` to install
the isolated dependencies while sharing the existing CUDA PyTorch. Then explicitly install the DiT,
Whisper, BigVGAN, and CAMPPlus files with `python scripts/download_models.py seed-vc`. This upstream
repository is archived; review its GPL-3.0 code/weights terms before redistribution. Research notes:
[docs/reference-voice-research.md](docs/reference-voice-research.md).

### Enable legacy RVC checkpoint conversion
RVC uses an isolated Python 3.12 environment. Follow [docs/adapters/rvc.md](docs/adapters/rvc.md), add an
authorized `.pth` under `models/rvc/`, and configure `RVC_PYTHON` / `RVC_PROJECT_DIR`. The UI convert page
uses Seed-VC; RVC remains available at `POST /api/audio/convert-rvc`.

## Run on Windows (PowerShell)
Backend:
    if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
    .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    Copy-Item backend\.env.example backend\.env
    .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000

Frontend (second terminal):
    Copy-Item frontend\.env.example frontend\.env
    npm --prefix frontend install
    npm --prefix frontend run dev

Open http://localhost:5173. Install FFmpeg (e.g. `winget install Gyan.FFmpeg`) for deeper audio validation.
To expose Demucs and Piper in the UI, use the backend environment with their packages installed.
See `docs/adapters/piper.md` for voice setup and model licences.

## Tests
    .\.venv\Scripts\python.exe -m pytest backend\tests -q
    # Optional real-model smoke tests run automatically when the required models are installed.
(The audio-decoding test is skipped unless `ffmpeg` is available on PATH. Real audio processing requires
FFmpeg and FFprobe.)

## Deployment preparation

The current application is designed for local use and has no Render, Vercel, Docker, authentication,
or cloud-storage configuration. See [docs/deployment.md](docs/deployment.md) before exposing it to a
network or deploying it.

## Structure
Matches the spec. Extra: `backend/app/models/schemas.py` holds Pydantic schemas
(the spec's `models/` folder at the root is reserved for model weights).

## Security notes
Files are stored under server-generated IDs, filenames are sanitised, uploads are size-limited
and content-checked, CORS is limited to configured origins, files expire after FILE_TTL_HOURS.
