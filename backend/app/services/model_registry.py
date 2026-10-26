"""Reports which AI models are really usable right now. Nothing is faked."""
from fastapi import HTTPException

from app.models.schemas import ModelStatus
from app.services.adapters import demucs_adapter, kokoro_adapter, piper_adapter, rvc_adapter, seed_vc_adapter


def list_models() -> list[ModelStatus]:
    demucs_ok, demucs_detail = demucs_adapter.status()
    tts_ok, tts_detail = piper_adapter.status()
    kokoro_ok, kokoro_detail = kokoro_adapter.status()
    seed_ok, seed_detail = seed_vc_adapter.status()
    rvc_ok, rvc_detail = rvc_adapter.status()
    return [
        ModelStatus(id="demucs", name="Demucs", feature="Vocal separation",
                    available=demucs_ok, detail=demucs_detail),
        ModelStatus(id="tts", name="Piper", feature="Text-to-speech",
                    available=tts_ok, detail=tts_detail),
        ModelStatus(id="kokoro", name="Kokoro", feature="Text-to-speech",
                    available=kokoro_ok, detail=kokoro_detail),
        ModelStatus(id="seed-vc", name="Seed-VC", feature="Voice conversion",
                    available=seed_ok, detail=seed_detail),
        ModelStatus(id="rvc", name="RVC", feature="Checkpoint voice conversion",
                    available=rvc_ok, detail=rvc_detail),
    ]


def require_model(model_id: str) -> None:
    for m in list_models():
        if m.id == model_id and not m.available:
            raise HTTPException(status_code=503, detail=f"{m.name} is unavailable. {m.detail}")
