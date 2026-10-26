"""Local Kokoro-82M speech synthesis using explicitly installed model and voice files."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.services import audio_io, storage
from app.services.adapters import AdapterError

log = logging.getLogger("vocal-studio.kokoro")
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_FILENAME = "kokoro-v1_0.pth"
CONFIG_FILENAME = "config.json"
VOICE_RE = re.compile(r"^(af|am|bf|bm)_[a-z]+$")
LANGUAGE_LABELS = {"a": "American English", "b": "British English"}
GENDER_LABELS = {"f": "female", "m": "male"}
_pipeline: Any | None = None


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    path: Path


def model_dir() -> Path:
    return settings.models_dir / "kokoro"


def list_voices() -> list[Voice]:
    folder = model_dir() / "voices"
    if not folder.is_dir():
        return []
    voices = []
    for path in sorted(folder.glob("*.pt"), key=lambda item: item.stem.casefold()):
        if path.is_file() and path.stat().st_size > 0 and VOICE_RE.fullmatch(path.stem):
            accent, gender = path.stem[:2], path.stem[1]
            label = f"{LANGUAGE_LABELS[accent[0]]} - {path.stem[3:].replace('_', ' ').title()} ({GENDER_LABELS[gender]})"
            voices.append(Voice(id=path.stem, name=label, path=path.resolve()))
    return voices


def status() -> tuple[bool, str]:
    if not all(find_spec(package) for package in ("kokoro", "torch", "soundfile", "transformers", "misaki")):
        return False, "Kokoro dependencies are not installed. Install backend/requirements-kokoro.txt in the backend environment."
    root = model_dir()
    config_path = root / CONFIG_FILENAME
    checkpoint_path = root / MODEL_FILENAME
    if not config_path.is_file() or not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return False, "Kokoro model files are not installed. Run `python scripts/download_models.py kokoro`."
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return False, "Kokoro model configuration is invalid. Reinstall it with `python scripts/download_models.py kokoro`."
    if not isinstance(config, dict) or not {"vocab", "n_token", "plbert"}.issubset(config):
        return False, "Kokoro model configuration is invalid. Reinstall it with `python scripts/download_models.py kokoro`."
    voices = list_voices()
    if not voices:
        return False, "No supported Kokoro English voice packs are installed. Run `python scripts/download_models.py kokoro --voice af_heart`."
    return True, f"Kokoro-82M is ready with {len(voices)} installed English voice(s)."


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KModel, KPipeline

        root = model_dir()
        model = KModel(
            repo_id=MODEL_REPO,
            config=str(root / CONFIG_FILENAME),
            model=str(root / MODEL_FILENAME),
        ).cpu().eval()
        _pipeline = KPipeline(lang_code="a", repo_id=MODEL_REPO, model=model)
    return _pipeline


def _device(torch) -> str:
    selected = settings.kokoro_device
    if selected == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise AdapterError("Kokoro is configured for CUDA, but CUDA is not available.")
    return selected


def synthesize(job, text: str, voice_id: str) -> dict[str, Path]:
    if not text.strip() or len(text.strip()) > settings.max_tts_characters:
        raise AdapterError(f"Text must contain 1 to {settings.max_tts_characters} non-whitespace characters.")
    voice = next((item for item in list_voices() if item.id == voice_id), None)
    if voice is None:
        raise AdapterError("The selected Kokoro voice is not installed.")
    available, detail = status()
    if not available:
        raise AdapterError(f"Kokoro is unavailable. {detail}")

    import torch

    pipeline = _get_pipeline()
    device = _device(torch)
    job.message = f"Generating speech with Kokoro ({voice.name})"
    job.device = device
    chunks: list[np.ndarray] = []
    try:
        pipeline.model.to(device)
        with torch.inference_mode():
            for index, (_, _, audio) in enumerate(pipeline(text.strip(), voice=str(voice.path)), start=1):
                job.message = f"Synthesizing speech segment {index}"
                if audio is not None:
                    chunks.append(audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio))
    except Exception:
        log.exception("Kokoro speech synthesis failed for job %s", job.job_id)
        raise AdapterError("Kokoro speech generation failed. Check the backend logs for details.") from None
    finally:
        pipeline.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not chunks:
        raise AdapterError("Kokoro did not generate any speech audio.")
    audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
    out_path = storage.job_output_dir(job.job_id) / "speech.wav"
    audio_io.write_wav(out_path, audio[None, :], 24000)
    try:
        audio_io.verify_wav(out_path, require_signal=True)
    except audio_io.AudioIOError as exc:
        out_path.unlink(missing_ok=True)
        raise AdapterError(f"Kokoro finished but the audio is not usable: {exc}") from exc
    return {"speech": out_path}
