# Adapter: vocal separation (Demucs)

| Item | Detail |
|---|---|
| Code | `backend/app/services/adapters/demucs_adapter.py` (API side) and `demucs_runner.py` (separate process) |
| Official source | https://github.com/facebookresearch/demucs (archived by Meta). We use the PyPI package `demucs` 4.1.0. |
| Exact model | `htdemucs` (Hybrid Transformer Demucs, 4 stems) from Hugging Face repo `adefossez/HTDemucs`, safetensors format. Vocals are output as-is; drums + bass + other are summed into "instrumental". |
| Code licence | MIT (package metadata of demucs 4.1.0) |
| Weights licence | **Unresolved.** Public discussion on the model's Hugging Face page quotes the author saying the pretrained weights are not covered by the code's MIT licence and are provided for scientific purposes. Use for personal/portfolio work only until you get written clarification from the authors. |
| Install | Install PyTorch (CUDA build) first, then `pip install -r requirements-ai.txt`, then `python scripts/download_models.py demucs` (asks first). |
| Hardware | NVIDIA GPU through CUDA, or CPU. `DEMUCS_DEVICE=auto` picks the GPU if PyTorch sees one. If the GPU runs out of memory the job retries on the CPU and says so. |
| Input | Any file FFmpeg can decode (WAV, MP3, FLAC, OGG, M4A), up to `MAX_UPLOAD_MB` and `MAX_AUDIO_MINUTES`. |
| Output | `vocals.wav` and `instrumental.wav`: stereo, 44.1 kHz, 16-bit PCM. Each file is re-opened and checked after writing. |
| Safety | Weights load only from `models/hf` (offline mode on, bypassing Demucs's automatic fallback to Meta's AWS server). Safetensors files are not executable pickles. |
| Limitations | Some vocal/instrument bleed is normal. Re-encoding to 16-bit is not bit-exact. Long tracks take a while. Not tested here on a GPU: first real run is on your machine. |
