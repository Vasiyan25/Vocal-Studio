# Adapter: text-to-speech (Piper)

| Item | Detail |
|---|---|
| Code | `backend/app/services/adapters/piper_adapter.py` |
| Official source | https://github.com/OHF-Voice/piper1-gpl (PyPI `piper-tts` 1.8.0). The earlier MIT repo `rhasspy/piper` was archived in October 2025. |
| Exact model | Voice `en_US-lessac-medium` (1 speaker, 22,050 Hz) from https://huggingface.co/rhasspy/piper-voices |
| Engine licence | GPL-3.0-or-later (package metadata). Run as a separate process (`python -m piper`), never imported. If you ever distribute this project, take legal advice on GPL obligations. |
| Voice licence | Separate from the engine. The model card points to the Blizzard 2013 Lessac dataset licence: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html . I have not read its terms for you; read them before any commercial use. |
| Install | `pip install -r requirements-ai.txt`, then `python scripts/download_models.py piper` (asks first). |
| Hardware | CPU only (onnxruntime). |
| Input | Text, 1 to 2000 characters. |
| Output | `speech.wav`: mono 16-bit PCM. Verified to be a valid, non-silent WAV after generation. |
| Limitations | One English voice, no style or emotion control, odd reading of some numbers and abbreviations. |
