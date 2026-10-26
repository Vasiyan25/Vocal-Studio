# Reference-conditioned voice conversion research notes

Date: 2026-10-24  
Goal: source audio + target speaker reference → same content, target-like identity.  
Hardware target: Windows, Python 3.12, NVIDIA RTX 4050 Laptop 6 GB VRAM.

## Options reviewed

### Seed-VC (selected)
- Repo: https://github.com/Plachtaa/seed-vc
- Interface: `inference.py --source <wav> --target <reference> --output <dir>`
- Zero-shot; no per-speaker training for basic use
- Preset used: `seed-uvit-whisper-small-wavenet` /  
  `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`
- VRAM: offline VC practical from ~4 GB; 6 GB OK with fp16 for short speech
- License: GPL-3.0 (code + HF weights card)
- Python: upstream suggests 3.10; isolated 3.12 env used in this project

### OpenVoice (not selected as primary)
- ToneColorConverter can rewrite source audio tone color given source/target SE
- Stronger product story around MeloTTS → converter (TTS cloning) than pure STS
- MIT license is friendlier for redistribution
- Rejected as primary because Seed-VC’s documented STS CLI matches the product UI directly

### RVC (kept as legacy `/api/audio/convert-rvc`)
- Checkpoint-conditioned; does **not** accept arbitrary reference audio as the speaker model
- Already integrated for authorized `.pth` voices

### XTTS / similar TTS cloners
- Text → speech with a reference speaker
- Not equivalent to converting an existing source recording’s content

### Pitch shift / EQ
- Explicitly out of scope (not speaker conversion)

## Decision
Implement Seed-VC in an isolated venv, dual-upload UI, and `POST /api/audio/convert`
with `file_id` + `reference_file_id`. Preserve Demucs, Piper, and RVC adapters.
