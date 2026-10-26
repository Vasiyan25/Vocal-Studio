"""Small audio helpers that only need FFmpeg, numpy and the standard library.

Used by the AI adapters to decode input, write WAV output and, importantly,
to check that an output file really exists and is a playable WAV.
"""
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.core.config import settings


class AudioIOError(Exception):
    """A readable problem with an audio file."""


def decode(path: Path, samplerate: int, channels: int) -> np.ndarray:
    """Decode any FFmpeg-readable file to float32 audio shaped (channels, samples)."""
    exe = shutil.which(settings.ffmpeg_path)
    if not exe:
        raise AudioIOError("FFmpeg was not found. Install it and restart the backend.")
    proc = subprocess.run(
        [exe, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(samplerate), "pipe:1"],
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise AudioIOError("FFmpeg could not decode this audio file.")
    data = np.frombuffer(proc.stdout, dtype="<f4")
    frames = len(data) // channels
    return data[: frames * channels].reshape(frames, channels).T.copy()


def write_wav(path: Path, audio: np.ndarray, samplerate: int) -> None:
    """Write float audio shaped (channels, samples) as 16-bit PCM WAV, avoiding clipping."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())


def verify_wav(path: Path, require_signal: bool = False) -> float:
    """Return the duration in seconds, or raise AudioIOError if the WAV is missing/invalid/empty."""
    try:
        with wave.open(str(path), "rb") as w:
            frames, rate, width = w.getnframes(), w.getframerate(), w.getsampwidth()
            raw = w.readframes(frames)
    except (wave.Error, EOFError, FileNotFoundError, OSError) as e:
        raise AudioIOError(f"The output file is missing or is not a valid WAV ({e}).")
    if frames == 0 or rate == 0:
        raise AudioIOError("The output audio is empty.")
    if require_signal and width == 2 and not np.frombuffer(raw, dtype="<i2").any():
        raise AudioIOError("The output audio is completely silent.")
    return frames / rate
