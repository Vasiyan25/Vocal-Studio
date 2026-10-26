"""Text-to-speech with Piper.

Model repository : https://github.com/OHF-Voice/piper1-gpl (PyPI `piper-tts` 1.8.0). The original
                   rhasspy/piper (MIT) was archived in October 2025.
Exact model      : voice `en_US-lessac-medium` (one speaker, 22,050 Hz) from https://huggingface.co/rhasspy/piper-voices
Licence          : engine GPL-3.0-or-later (changed from MIT). We run it as a separate process (`python -m piper`)
                   and never import it. The voice has its OWN licence: its dataset (Blizzard 2013, Lessac) is covered
                   by https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html . Read it
                   before any commercial use. This is not legal advice.
Install          : `pip install -r requirements-ai.txt`; download the voice with scripts/download_models.py piper.
Hardware         : CPU (onnxruntime). No GPU needed.
Input / output   : plain text (up to 2000 characters) in; mono 16-bit WAV out (speech.wav).
Limitations      : one English voice; no emotion or style control; numbers and abbreviations may be read oddly.
"""
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from app.core.config import ROOT, settings
from app.services import audio_io, storage
from app.services.adapters import AdapterError


@dataclass(frozen=True)
class VoiceModel:
    id: str
    onnx_path: Path
    config_path: Path


def default_voice_id() -> str | None:
    if settings.piper_voice:
        return settings.piper_voice
    voices = list_voices()
    return voices[0].id if voices else None


def list_voices() -> list[VoiceModel]:
    folder = settings.models_dir / "piper"
    if not folder.is_dir():
        return []
    voices: list[VoiceModel] = []
    for onnx in sorted(folder.glob("*.onnx"), key=lambda path: path.name.casefold()):
        if not onnx.is_file() or onnx.stat().st_size == 0:
            continue
        config = onnx.with_name(f"{onnx.name}.json")
        if config.exists() and config.stat().st_size > 0:
            voices.append(VoiceModel(id=onnx.stem, onnx_path=onnx.resolve(), config_path=config.resolve()))
    if not voices and settings.piper_voice:
        onnx = folder / f"{settings.piper_voice}.onnx"
        if onnx.exists() and onnx.stat().st_size > 0:
            config = onnx.with_name(f"{onnx.name}.json")
            if config.exists():
                voices.append(VoiceModel(id=settings.piper_voice, onnx_path=onnx.resolve(), config_path=config.resolve()))
    return voices


def find_voice(voice_id: str | None) -> VoiceModel | None:
    if not voice_id:
        default = default_voice_id()
        return next((voice for voice in list_voices() if voice.id == default), None)
    return next((voice for voice in list_voices() if voice.id == voice_id), None)


def voice_paths(voice_id: str | None = None) -> tuple[Path, Path]:
    voice = find_voice(voice_id) or find_voice(default_voice_id())
    if voice is not None:
        return voice.onnx_path, voice.config_path
    onnx = settings.models_dir / "piper" / f"{settings.piper_voice}.onnx"
    return onnx, onnx.with_name(onnx.name + ".json")


def status() -> tuple[bool, str]:
    if not find_spec("piper"):
        return False, "Not installed. With the backend venv active run: pip install -r requirements-ai.txt"
    voices = list_voices()
    if not voices:
        return False, "No Piper voice files were found under models/piper. Download a compatible .onnx and .json pair with `python scripts/download_models.py piper` before generating speech."
    default = default_voice_id()
    if default is None:
        return False, "No Piper voice is configured. Add a voice under models/piper and restart the backend."
    return True, f"Piper voice {default} is ready (runs on the CPU)."


def synthesize(job, text: str, voice_id: str | None = None) -> dict[str, Path]:
    voices = list_voices()
    if not voices:
        raise AdapterError("Text-to-speech is unavailable. No Piper voice files were found under models/piper.")
    voice = find_voice(voice_id) or find_voice(default_voice_id())
    if voice is None:
        raise AdapterError("The selected Piper voice is not installed. Select a valid voice and try again.")
    onnx, cfg = voice.onnx_path, voice.config_path
    out_path = storage.job_output_dir(job.job_id) / "speech.wav"
    job.message = f"Generating speech with {voice.id}"
    job.device = "cpu"
    line = " ".join(text.split())
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(onnx), "-c", str(cfg), "-f", str(out_path)],
            input=line + "\n", capture_output=True, text=True, encoding="utf-8",
            timeout=min(settings.job_timeout_s, 300), cwd=settings.backend_dir,
        )
    except subprocess.TimeoutExpired:
        raise AdapterError("Speech generation took too long and was stopped.")
    if proc.returncode != 0:
        raise AdapterError("Speech generation failed: " + (proc.stderr.replace(str(ROOT), "<project>").strip()[-500:] or f"exit code {proc.returncode}"))
    try:
        audio_io.verify_wav(out_path, require_signal=True)
    except audio_io.AudioIOError as e:
        raise AdapterError(f"Speech generation finished but the audio is not usable: {e}")
    return {"speech": out_path}
