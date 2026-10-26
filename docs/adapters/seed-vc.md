# Seed-VC reference-conditioned voice conversion

Vocal Studio uses the official Seed-VC offline speech-conversion workflow: a source recording and a
separate target-speaker reference are passed as `--source` and `--target`. This is zero-shot
reference-conditioned conversion; it does not use an RVC speaker checkpoint or pitch shifting.

Upstream: [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc)  
Model repository: [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)  
Selected preset: `seed-uvit-whisper-small-wavenet`, speech VC at 22.05 kHz  
License: GPL-3.0 code and checkpoint. Review the licence before redistribution or commercial use.

## Upstream status and dependencies

GitHub marks the upstream repository **archived**; its last push was 20 April 2025. This project
uses the checked-out upstream implementation and its inference requirements, with small local changes
to preserve the configured cache and make all inference offline after setup. Treat upstream
compatibility and maintenance as a deployment risk.

The inference requirements include `transformers==4.46.3`, `numpy==1.26.4`, librosa, FunASR, and
other audio packages. They are isolated in `backend/.venv-seed-vc`. The setup script shares the
already-installed main environment PyTorch/Torchaudio via `--system-site-packages`, so it does not
download or replace a second CUDA PyTorch wheel. It is necessary that the main `.venv` has a
working CUDA PyTorch and Torchaudio install before using this setup. The Seed-VC combination with
your installed Torch version still requires an actual inference run to certify.

## Install the runtime (Windows PowerShell)

Run from the project root. Review and explicitly confirm the dependency install:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_seed_vc.ps1
```

The script checks the existing CUDA PyTorch, creates or adjusts the Python 3.12 isolated venv to
reuse it, and installs `backend\requirements-seed-vc.txt`. It does not install model files.
Configuration in `backend\.env`:

```env
SEED_VC_PYTHON=backend/.venv-seed-vc/Scripts/python.exe
SEED_VC_PROJECT_DIR=backend/seed-vc-project
SEED_VC_DEVICE=auto
SEED_VC_DIFFUSION_STEPS=25
SEED_VC_FP16=true
MAX_REFERENCE_SECONDS=30
```

`SEED_VC_DEVICE` accepts `auto`, `cuda`, or `cpu`. On a 6 GB RTX 4050, start with short speech
clips, fp16, and 10–25 diffusion steps. Reduce steps or use CPU if CUDA runs out of memory.

## Explicit model and auxiliary-asset download

The complete inference needs more than the DiT checkpoint: the matching config, Whisper-small
encoder, CAMPPlus speaker encoder, and BigVGAN 22 kHz vocoder. Inference is configured to run
offline; it will not silently download missing model assets. From the project root, explicitly run:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py seed-vc
```

The script reports the Hugging Face size estimate and asks for confirmation before downloading.
It installs the exact official files:

```text
models/seed-vc/
  DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
  config_dit_mel_seed_uvit_whisper_small_wavenet.yml
  hf_cache/
    models--openai--whisper-small/
    models--nvidia--bigvgan_v2_22khz_80band_256x/
    models--funasr--campplus/
```

The current official DiT checkpoint is 440,312,082 bytes with SHA-256
`8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720`; the setup script verifies
both before proceeding.

Do not rename or substitute checkpoints. The Seed-VC checkpoint repository is GPL-3.0; auxiliary
models have separate licences and terms. Review each upstream model card before commercial use or
redistribution. These large files are not included in Git or the frontend.

After setup, restart the backend. `GET /api/models` reports the engine available only when the
runtime, exact DiT checkpoint/config, and auxiliary cached models are present. Missing files return
an unavailable status or a real conversion error; there is no success-shaped fallback.

## API, test, and limitations

`POST /api/audio/convert` takes distinct uploaded IDs:

```json
{ "file_id": "<source upload id>", "reference_file_id": "<target sample upload id>" }
```

The response is the normal queued job. Poll `GET /api/jobs/{job_id}` and use the existing
`/stream` and `/download` job routes. Source and target audio are content-validated before the job
is queued; the reference duration limit defaults to 30 seconds.

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

The suite includes a real conversion smoke test when every runtime and model asset is installed;
otherwise it is skipped with the setup reason. Real output quality depends on clean speech and can
vary for noisy music, cross-language input, and long clips. The selected checkpoint is for speech,
not the higher-memory singing/f0 model.
