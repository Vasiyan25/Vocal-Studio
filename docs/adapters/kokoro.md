# Kokoro-82M text-to-speech

Vocal Studio uses the official Kokoro Python inference package and the
[`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) model. The TTS page lets you
select Kokoro or Piper and lists only voice packs installed locally. The voice ID sent to the API is
the original Kokoro ID (for example, `af_heart`); the displayed label adds language and voice type.

## Compatibility and licensing

- The upstream `kokoro` 0.9.4 package supports Python 3.10–3.12 and depends on PyTorch, Transformers,
  Misaki, NumPy, Hugging Face Hub, and Loguru.
- Kokoro model weights are listed as **Apache-2.0** on the official model card. Individual voice
  training-data attributions differ; review the model card and `VOICES.md` before redistribution or
  commercial use. Package, dependency, and voice/data terms are not one blanket licence.
- English is the supported UI set here. American and British English voice packs whose IDs begin
  `af_`, `am_`, `bf_`, or `bm_` are selectable when their `.pt` files are installed.
- English G2P uses Misaki. Install eSpeak-NG from its official release for the best out-of-vocabulary
  pronunciation fallback; inference still uses the Kokoro model.

## Install and explicitly download model files (Windows PowerShell)

Run from the project root. These commands use the existing Python environment and do not request a
new PyTorch/CUDA wheel:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-kokoro.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts\download_models.py kokoro --voice af_heart
```

The model setup command prints the Hugging Face-reported download size and asks for confirmation.
Add any number of other official English voices by repeating `--voice`, for example:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py kokoro --voice af_heart --voice am_michael --voice bf_emma
```

Model files are installed under `models\kokoro\` and never enter the frontend build. The backend
loads one model lazily, reuses it between TTS requests, and returns it to CPU after each GPU job so
Demucs and Seed-VC can use the single-worker queue without leaving Kokoro resident on the GPU.
`KOKORO_DEVICE` accepts `auto`, `cuda`, or `cpu`; `MAX_TTS_CHARACTERS` defaults to 2000.

## API and test

- `GET /api/models` reports Kokoro readiness.
- `GET /api/tts/voices/kokoro` returns `{id, name}` entries for installed voice packs.
- `POST /api/audio/tts` accepts `{"text":"Hello","model_id":"kokoro","voice_id":"af_heart"}` and
  returns the normal job response. Omit `voice_id` to select the first installed voice.
- `POST /api/audio/tts` without a `model_id` remains backward-compatible with Piper (`tts`).

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

The test suite includes a real Kokoro generation smoke test when Kokoro and model assets are present;
otherwise that test is skipped with the missing setup detail.
