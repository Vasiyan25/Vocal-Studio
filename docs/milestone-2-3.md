# Milestones 2 and 3: separation and text-to-speech

## What was verified, and where
- Verified in the build sandbox (Linux, no GPU, no model weights): API, job pipeline, polling, streaming, downloads, error handling,
  timeouts and audio file verification, using a fake runner for process handling (`backend/tests`).
- Verified against the real package sources (demucs 4.1.0, piper-tts 1.8.0): the function names, CLI flags and model attributes the adapters use.
- NOT verified until run on your machine: real Demucs on the RTX 4050, real Piper audio, and the browser flow with real models.

## Design choices
- Jobs run one at a time on a single background worker so two heavy jobs never compete for 6 GB of GPU memory.
- Demucs runs in a subprocess: GPU memory is freed afterwards and a crash cannot take down the API.
- No automatic downloads. `scripts/download_models.py` shows source, licence note and size, then waits for "yes".
- Output files live in `storage/outputs/<job_id>/` and are only reachable through `/api/jobs/{id}/stream|download?stem=`.
