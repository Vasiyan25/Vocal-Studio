from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]  # vocal-studio/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / "backend" / ".env", extra="ignore")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 50
    file_ttl_hours: int = 24
    ffprobe_path: str = "ffprobe"
    ffmpeg_path: str = "ffmpeg"
    backend_dir: Path = ROOT / "backend"
    models_dir: Path = ROOT / "models"
    demucs_device: Literal["auto", "cuda", "cpu"] = "auto"
    rvc_python: Path | None = None
    rvc_project_dir: Path | None = None
    rvc_device: Literal["auto", "cuda", "cpu"] = "auto"
    # Isolated Seed-VC zero-shot runtime (reference-conditioned speech-to-speech).
    seed_vc_python: Path | None = None
    seed_vc_project_dir: Path | None = None
    seed_vc_device: Literal["auto", "cuda", "cpu"] = "auto"
    seed_vc_diffusion_steps: int = 25
    seed_vc_fp16: bool = True
    max_reference_seconds: int = 30
    kokoro_device: Literal["auto", "cuda", "cpu"] = "auto"
    piper_voice: str = "en_US-lessac-medium"
    max_tts_characters: int = Field(default=2000, ge=1, le=2000)
    job_timeout_s: int = 1800
    max_audio_minutes: int = 20
    upload_dir: Path = ROOT / "storage" / "uploads"
    output_dir: Path = ROOT / "storage" / "outputs"
    allowed_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
