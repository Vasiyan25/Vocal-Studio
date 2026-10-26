"""Offline voice conversion through the official RVC project's inference CLI."""
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import ROOT, settings
from app.services import audio_io, storage
from app.services.adapters import AdapterError

VOICE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
VOICE_ID_RE = re.compile(VOICE_ID_PATTERN)
MAX_LOG_BYTES = 16000
RUNTIME_CHECK_TIMEOUT_S = 180
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


@dataclass(frozen=True)
class VoiceModel:
    id: str
    model_path: Path
    index_path: Path | None


def runtime_python() -> Path | None:
    configured = settings.rvc_python
    if configured is None:
        return None
    return configured if configured.is_absolute() else ROOT / configured


def runtime_project() -> Path | None:
    configured = settings.rvc_project_dir
    if configured is None:
        return None
    return configured if configured.is_absolute() else ROOT / configured


def list_voices() -> list[VoiceModel]:
    """List non-empty, regular checkpoint files; never load pickle data for discovery."""
    folder = settings.models_dir / "rvc"
    if not folder.is_dir():
        return []
    voices = []
    for model_path in sorted(folder.glob("*.pth"), key=lambda path: path.name.casefold()):
        voice_id = model_path.stem
        if not VOICE_ID_RE.fullmatch(voice_id) or not model_path.is_file() or model_path.is_symlink():
            continue
        if model_path.stat().st_size == 0:
            continue
        index_path = model_path.with_suffix(".index")
        if (
            index_path.is_file()
            and not index_path.is_symlink()
            and index_path.stat().st_size > 0
        ):
            resolved_index = index_path.resolve()
        else:
            resolved_index = None
        voices.append(VoiceModel(
            id=voice_id,
            model_path=model_path.resolve(),
            index_path=resolved_index,
        ))
    return voices


def find_voice(voice_id: str) -> VoiceModel | None:
    return next((voice for voice in list_voices() if voice.id == voice_id), None)


def _offline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(OFFLINE_ENV)
    if settings.rvc_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _runtime_status() -> tuple[bool, str, str | None]:
    executable = runtime_python()
    project = runtime_project()
    if executable is None or project is None:
        return False, "Configure RVC_PYTHON and RVC_PROJECT_DIR in backend/.env.", None
    if not executable.is_file():
        return False, "The configured RVC Python environment is missing. See docs/adapters/rvc.md.", None
    if not (project / "infer" / "cli.py").is_file():
        return False, "The official RVC project checkout is missing. See docs/adapters/rvc.md.", None
    if not shutil.which(settings.ffmpeg_path) or not shutil.which(settings.ffprobe_path):
        return False, "FFmpeg and FFprobe are required for RVC conversion and must be available on PATH.", None

    hubert = project / "assets" / "hubert_base"
    has_unsharded_hubert_weights = any(
        (hubert / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    has_sharded_hubert_weights = any(hubert.glob("model-*-of-*.safetensors")) or any(
        hubert.glob("pytorch_model-*-of-*.bin")
    )
    if (
        not (hubert / "config.json").is_file()
        or not (hubert / "preprocessor_config.json").is_file()
        or not (has_unsharded_hubert_weights or has_sharded_hubert_weights)
    ):
        return False, "The local HuBERT inference assets are missing under the RVC project assets/hubert_base folder.", None
    if not (project / "assets" / "rmvpe" / "rmvpe.pt").is_file():
        return False, "The local RMVPE pitch model is missing under the RVC project assets/rmvpe folder.", None

    check = (
        "import importlib.metadata as metadata, sys; "
        "packages=('numba','llvmlite','numpy','librosa'); "
        "print('RVC_DEPENDENCIES='+'; '.join(f'{name}={metadata.version(name)}' "
        "for name in packages), flush=True); "
        "import torch, numpy, faiss, librosa, soundfile, transformers, numba, llvmlite; "
        "import numba._devicearray; "
        "from infer.vc.modules import VC; "
        "assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'; "
        "assert torch.__version__.startswith('2.7.1'), 'PyTorch 2.7.1 is required'; "
        "print('RVC_DEVICE=' + ('cuda' if torch.cuda.is_available() else 'cpu'))"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", check],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RUNTIME_CHECK_TIMEOUT_S,
            cwd=project,
            env=_offline_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "Checking the RVC runtime timed out.", None
    except OSError:
        return False, "Could not start the configured RVC Python environment.", None
    if result.returncode != 0:
        diagnostic = (result.stdout + "\n" + result.stderr).strip()
        for private_path in (str(ROOT), str(project), str(executable)):
            diagnostic = diagnostic.replace(private_path, "<private-path>")
        diagnostic = diagnostic[-1600:]
        suffix = f" Details: {diagnostic}" if diagnostic else ""
        return False, (
            "RVC runtime/dependency check failed. Check the reported Numba, llvmlite, and "
            "NumPy versions and the import error; Windows application-control policy may "
            "block native extensions."
            f"{suffix}"
        ), None
    device_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith("RVC_DEVICE=")),
        "RVC_DEVICE=cpu",
    )
    device = device_line.removeprefix("RVC_DEVICE=")
    if settings.rvc_device == "cuda" and device != "cuda":
        return False, "RVC is configured for CUDA, but CUDA is unavailable in its isolated environment.", None
    return True, "Official RVC inference runtime and local inference assets are ready.", device


def status() -> tuple[bool, str]:
    runtime_ok, detail, device = _runtime_status()
    if not runtime_ok:
        return False, detail
    voices = list_voices()
    if not voices:
        return False, "No non-empty voice checkpoints found. Add an authorized .pth file under models/rvc/."
    selected_device = device if settings.rvc_device == "auto" else settings.rvc_device
    return True, (
        f"RVC runtime is ready with {len(voices)} checkpoint(s) on {selected_device}. "
        "Checkpoint compatibility is checked during conversion."
    )


def _capture_output(stream, tail: bytearray) -> None:
    while chunk := stream.read(4096):
        tail.extend(chunk)
        if len(tail) > MAX_LOG_BYTES:
            del tail[:-MAX_LOG_BYTES]


def _failure_message(log: str, returncode: int, hidden_paths: tuple[Path, ...] = ()) -> str:
    lowered = log.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return "RVC ran out of GPU memory. Try a shorter vocal input or set RVC_DEVICE=cpu."
    cleaned = log.replace(str(ROOT), "<project>").strip()
    for path in hidden_paths:
        cleaned = cleaned.replace(str(path), "<private-path>")
    return "Voice conversion failed: " + (cleaned or f"RVC runner exited with code {returncode}.")


def convert(job, source: Path, voice_id: str) -> dict[str, Path]:
    runtime_ok, detail, detected_device = _runtime_status()
    if not runtime_ok:
        raise AdapterError(f"RVC is unavailable. {detail}")
    voice = find_voice(voice_id)
    if voice is None:
        raise AdapterError("The selected voice checkpoint is no longer available. Refresh the page and try again.")
    if settings.rvc_device == "cuda" and detected_device != "cuda":
        raise AdapterError("RVC is configured for CUDA, but CUDA is unavailable in its isolated environment.")

    device = detected_device if settings.rvc_device == "auto" else settings.rvc_device
    output_path = storage.job_output_dir(job.job_id) / "converted.wav"
    command = [
        str(runtime_python()),
        str(ROOT / "backend" / "app" / "services" / "adapters" / "rvc_runner.py"),
        "--rvc-root", str(runtime_project()),
        "--models-dir", str(settings.models_dir / "rvc"),
        "--voice-id", voice.id,
        "--input", str(source.resolve()),
        "--output", str(output_path),
        "--device", device,
    ]
    job.message = f"Converting voice with {voice.id}"
    job.device = device

    try:
        process = subprocess.Popen(
            command,
            cwd=runtime_project(),
            env=_offline_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        output_path.unlink(missing_ok=True)
        raise AdapterError("Could not start the RVC runtime. Check RVC_PYTHON and RVC_PROJECT_DIR in backend/.env.")
    tail = bytearray()
    reader = threading.Thread(target=_capture_output, args=(process.stdout, tail), daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=settings.job_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join()
        output_path.unlink(missing_ok=True)
        raise AdapterError(
            f"Voice conversion took longer than {settings.job_timeout_s} seconds and was stopped."
        )
    finally:
        if process.stdout:
            process.stdout.close()
    reader.join()
    output = bytes(tail).decode("utf-8", errors="replace")[-4000:]

    if returncode != 0:
        output_path.unlink(missing_ok=True)
        raise AdapterError(_failure_message(
            output, returncode, (voice.model_path, source, runtime_project() or ROOT)
        ))
    try:
        audio_io.verify_wav(output_path, require_signal=True)
    except audio_io.AudioIOError as e:
        output_path.unlink(missing_ok=True)
        raise AdapterError(f"Voice conversion finished but the audio is not usable: {e}")
    return {"converted": output_path}
