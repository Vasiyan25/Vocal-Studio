from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    duration_seconds: Optional[float] = None
    format: str


class SeparateRequest(BaseModel):
    file_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    model_id: Literal["tts", "piper", "kokoro"] = "tts"

    @field_validator("text")
    @classmethod
    def text_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value


class TtsVoice(BaseModel):
    id: str
    name: str


class ConvertRequest(BaseModel):
    """Reference-conditioned conversion: source speech + target speaker sample."""
    file_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    reference_file_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class ConvertRvcRequest(BaseModel):
    """Legacy checkpoint-conditioned RVC conversion (authorized .pth voices)."""
    file_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    voice_model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class ModelStatus(BaseModel):
    id: str
    name: str
    feature: str
    available: bool
    detail: str


class JobStatus(BaseModel):
    job_id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    device: Optional[str] = None
    outputs: list[str] = []
    error: Optional[str] = None
