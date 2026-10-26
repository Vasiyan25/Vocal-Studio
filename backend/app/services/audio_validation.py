"""Verify that an upload really is audio, not just that it has an audio extension."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.config import settings


@dataclass
class AudioInfo:
    format: str
    duration: Optional[float]


def _magic_ok(head: bytes, ext: str) -> bool:
    if ext == ".wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == ".mp3":
        return head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
    if ext == ".flac":
        return head[:4] == b"fLaC"
    if ext == ".ogg":
        return head[:4] == b"OggS"
    if ext == ".m4a":
        return head[4:8] == b"ftyp"
    return False


def ffprobe_available() -> bool:
    return shutil.which(settings.ffprobe_path) is not None


def verify_audio(path: Path, ext: str) -> AudioInfo:
    """Raises ValueError with a user-readable message if the file is not valid audio."""
    with open(path, "rb") as f:
        head = f.read(16)
    if not _magic_ok(head, ext):
        raise ValueError(f"The file contents do not look like a {ext[1:].upper()} audio file.")

    duration = None
    if ffprobe_available():
        try:
            out = subprocess.run(
                [settings.ffprobe_path, "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=20, check=True,
            )
            data = json.loads(out.stdout or "{}")
        except (subprocess.SubprocessError, json.JSONDecodeError):
            raise ValueError("FFmpeg could not read this file. It may be corrupted.")
        if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
            raise ValueError("No audio stream was found in this file.")
        try:
            duration = float(data["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            duration = None
    return AudioInfo(format=ext[1:], duration=duration)
