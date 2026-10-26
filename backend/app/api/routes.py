import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import (ConvertRequest, ConvertRvcRequest, JobStatus, ModelStatus,
                                SeparateRequest, TtsRequest, TtsVoice, UploadResponse)
from app.services import audio_validation, jobs, storage
from app.services.adapters import demucs_adapter, kokoro_adapter, piper_adapter, rvc_adapter, seed_vc_adapter
from app.services.model_registry import list_models, require_model

router = APIRouter(prefix="/api")

MEDIA_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
               ".ogg": "audio/ogg", ".m4a": "audio/mp4"}


@router.get("/health")
def health():
    return {"status": "ok", "ffprobe": audio_validation.ffprobe_available(),
            "max_upload_mb": settings.max_upload_mb}


@router.get("/models", response_model=list[ModelStatus])
def models():
    return list_models()


@router.get("/voices", response_model=list[str])
def voices():
    return [voice.id for voice in rvc_adapter.list_voices()]


@router.get("/tts/voices", response_model=list[str])
def tts_voices():
    return [voice.id for voice in piper_adapter.list_voices()]


@router.get("/tts/voices/{model_id}", response_model=list[TtsVoice])
def tts_voices_for_model(model_id: str):
    if model_id in {"tts", "piper"}:
        return [TtsVoice(id=voice.id, name=voice.id.replace("_", " ")) for voice in piper_adapter.list_voices()]
    if model_id == "kokoro":
        return [TtsVoice(id=voice.id, name=voice.name) for voice in kokoro_adapter.list_voices()]
    raise HTTPException(status_code=404, detail="Unknown text-to-speech model.")


@router.post("/audio/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    storage.ensure_dirs()
    clean_name = storage.safe_filename(file.filename or "audio")
    ext = ("." + clean_name.rsplit(".", 1)[-1].lower()) if "." in clean_name else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}")

    file_id = storage.new_id()
    dest = storage.upload_path(file_id, ext)
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            info = audio_validation.verify_audio(dest, ext)
        except ValueError as e:
            raise HTTPException(400, str(e))
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return UploadResponse(file_id=file_id, filename=clean_name, size_bytes=size,
                          duration_seconds=info.duration, format=info.format)


@router.get("/audio/{file_id}/stream")
def stream_audio(file_id: str):
    path = storage.find_upload(file_id)
    if not path:
        raise HTTPException(404, "Audio file not found.")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"))


@router.post("/audio/separate", response_model=JobStatus, status_code=202)
def separate(req: SeparateRequest):
    src = storage.find_upload(req.file_id)
    if not src:
        raise HTTPException(404, "Audio file not found.")
    require_model("demucs")
    try:
        duration = audio_validation.verify_audio(src, src.suffix).duration
    except ValueError as e:
        raise HTTPException(400, str(e))
    if duration and duration > settings.max_audio_minutes * 60:
        raise HTTPException(413, f"Audio is longer than {settings.max_audio_minutes} minutes.")
    return _status(jobs.submit("separate", demucs_adapter.separate, src))


@router.post("/audio/tts", response_model=JobStatus, status_code=202)
def tts(req: TtsRequest):
    if len(req.text.strip()) > settings.max_tts_characters:
        raise HTTPException(
            status_code=422,
            detail=f"Text must be no longer than {settings.max_tts_characters} characters.",
        )
    model_id = "tts" if req.model_id == "piper" else req.model_id
    require_model(model_id)
    if model_id == "kokoro":
        voices = kokoro_adapter.list_voices()
        voice_id = req.voice_id or (voices[0].id if voices else None)
        if not voice_id or not any(voice.id == voice_id for voice in voices):
            raise HTTPException(status_code=404, detail="The selected Kokoro voice is not installed.")
        return _status(jobs.submit("tts", kokoro_adapter.synthesize, req.text, voice_id))

    available = piper_adapter.list_voices()
    configured = piper_adapter.default_voice_id()
    selected = req.voice_id or (configured if configured and any(v.id == configured for v in available) else (available[0].id if available else None))
    if req.voice_id and not any(v.id == req.voice_id for v in available):
        raise HTTPException(status_code=404, detail="The selected Piper voice is not installed.")
    return _status(jobs.submit("tts", piper_adapter.synthesize, req.text, selected))


@router.post("/audio/convert", response_model=JobStatus, status_code=202)
def convert(req: ConvertRequest):
    """Zero-shot reference-conditioned conversion (Seed-VC): source + target sample."""
    if req.file_id == req.reference_file_id:
        raise HTTPException(400, "Source audio and target voice sample must be different uploads.")
    src = storage.find_upload(req.file_id)
    if not src:
        raise HTTPException(404, "Source audio file not found.")
    reference = storage.find_upload(req.reference_file_id)
    if not reference:
        raise HTTPException(404, "Target voice sample not found.")
    require_model("seed-vc")
    try:
        src_info = audio_validation.verify_audio(src, src.suffix)
        ref_info = audio_validation.verify_audio(reference, reference.suffix)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if src_info.duration and src_info.duration > settings.max_audio_minutes * 60:
        raise HTTPException(413, f"Source audio is longer than {settings.max_audio_minutes} minutes.")
    if ref_info.duration and ref_info.duration > settings.max_reference_seconds:
        raise HTTPException(
            413,
            f"Target voice sample is longer than {settings.max_reference_seconds} seconds. "
            "Use a clean 3–30 second clip of the desired speaker.",
        )
    return _status(jobs.submit("convert", seed_vc_adapter.convert, src, reference))


@router.post("/audio/convert-rvc", response_model=JobStatus, status_code=202)
def convert_rvc(req: ConvertRvcRequest):
    """Legacy RVC conversion using an authorized local .pth checkpoint."""
    src = storage.find_upload(req.file_id)
    if not src:
        raise HTTPException(404, "Audio file not found.")
    require_model("rvc")
    if not rvc_adapter.find_voice(req.voice_model):
        raise HTTPException(404, "Voice model not found. Add an authorized .pth file under models/rvc and refresh.")
    try:
        duration = audio_validation.verify_audio(src, src.suffix).duration
    except ValueError as e:
        raise HTTPException(400, str(e))
    if duration and duration > settings.max_audio_minutes * 60:
        raise HTTPException(413, f"Audio is longer than {settings.max_audio_minutes} minutes.")
    return _status(jobs.submit("convert", rvc_adapter.convert, src, req.voice_model))


def _status(job) -> JobStatus:
    return JobStatus(job_id=job.job_id, kind=job.kind, status=job.status, message=job.message,
                     device=job.device, outputs=list(job.outputs), error=job.error)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return _status(job)


def _output_file(job_id: str, stem: str | None):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status != "completed" or not job.outputs:
        raise HTTPException(409, "Job has no finished output yet.")
    name = stem or next(iter(job.outputs))
    path = job.outputs.get(name)  # only names we created are accepted, never a client path
    if path is None:
        raise HTTPException(404, f"This job has no output called '{name}'. Available: {', '.join(job.outputs)}.")
    if not path.exists():
        raise HTTPException(410, "This file has expired and was cleaned up. Run the job again.")
    return job, name, path


@router.get("/jobs/{job_id}/stream")
def job_stream(job_id: str, stem: str | None = None):
    _, _, path = _output_file(job_id, stem)
    return FileResponse(path, media_type="audio/wav")


@router.get("/jobs/{job_id}/download")
def job_download(job_id: str, stem: str | None = None):
    job, name, path = _output_file(job_id, stem)
    return FileResponse(path, media_type="audio/wav", filename=f"{job.kind}-{name}.wav")
