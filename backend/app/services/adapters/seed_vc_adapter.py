"""Zero-shot reference-conditioned voice conversion via official Seed-VC inference.

Seed-VC takes a source utterance and a short target-speaker reference recording.
It does not use RVC-style per-speaker .pth training. Runtime is isolated in a
separate Python environment (see docs/adapters/seed-vc.md).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from app.core.config import ROOT, settings
from app.services import audio_io, storage
from app.services.adapters import AdapterError

MAX_LOG_BYTES = 16000
RUNTIME_CHECK_TIMEOUT_S = 180
OFFLINE_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

# Official Plachta/Seed-VC offline speech VC assets (verified from upstream README).
DIT_CHECKPOINT_NAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
DIT_CONFIG_NAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DIT_CHECKPOINT_SIZE = 440_312_082


def runtime_python() -> Path | None:
    configured = settings.seed_vc_python
    if configured is None:
        return None
    return configured if configured.is_absolute() else ROOT / configured


def runtime_project() -> Path | None:
    configured = settings.seed_vc_project_dir
    if configured is None:
        return None
    return configured if configured.is_absolute() else ROOT / configured


def weights_dir() -> Path:
    return settings.models_dir / "seed-vc"


def checkpoint_path() -> Path:
    return weights_dir() / DIT_CHECKPOINT_NAME


def config_path() -> Path:
    return weights_dir() / DIT_CONFIG_NAME


def hf_cache_dir() -> Path:
    return weights_dir() / "hf_cache"


def _cached_snapshot_has_files(repo_id: str, required_files: tuple[str, ...], weight_suffixes: tuple[str, ...]) -> bool:
    repo_dir = hf_cache_dir() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    for snapshot in repo_dir.iterdir() if repo_dir.is_dir() else ():
        if not all((snapshot / name).is_file() for name in required_files):
            continue
        if any(file.is_file() and file.suffix in weight_suffixes for file in snapshot.rglob("*")):
            return True
    return False


def _auxiliary_models_ready() -> bool:
    return (
        _cached_snapshot_has_files("openai/whisper-small", ("config.json",), (".safetensors", ".bin"))
        and _cached_snapshot_has_files(
            "nvidia/bigvgan_v2_22khz_80band_256x",
            ("config.json",),
            (".safetensors", ".bin", ".pt"),
        )
        and _cached_snapshot_has_files("funasr/campplus", ("campplus_cn_common.bin",), (".bin",))
    )


def _offline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(OFFLINE_ENV)
    cache = str(hf_cache_dir().resolve())
    env["HF_HUB_CACHE"] = cache
    env["HUGGINGFACE_HUB_CACHE"] = cache
    env["TRANSFORMERS_CACHE"] = cache
    env["SEED_VC_CACHE_DIR"] = cache
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if settings.seed_vc_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _runtime_status() -> tuple[bool, str, str | None]:
    executable = runtime_python()
    project = runtime_project()
    if executable is None or project is None:
        return False, "Configure SEED_VC_PYTHON and SEED_VC_PROJECT_DIR in backend/.env. See docs/adapters/seed-vc.md.", None
    if not executable.is_file():
        return False, "The configured Seed-VC Python environment is missing. See docs/adapters/seed-vc.md.", None
    if not (project / "inference.py").is_file():
        return False, "The official Seed-VC project checkout is missing. See docs/adapters/seed-vc.md.", None
    if not shutil.which(settings.ffmpeg_path) or not shutil.which(settings.ffprobe_path):
        return False, "FFmpeg and FFprobe are required for Seed-VC conversion and must be available on PATH.", None
    if not checkpoint_path().is_file() or checkpoint_path().stat().st_size != DIT_CHECKPOINT_SIZE:
        return False, "Required Seed-VC model files are missing.", None
    if not config_path().is_file() or config_path().stat().st_size == 0:
        return False, "Required Seed-VC model files are missing.", None
    if not _auxiliary_models_ready():
        return False, "Seed-VC auxiliary model files are not installed. Run `python scripts/download_models.py seed-vc`.", None

    check = (
        "import sys, torch, torchaudio, librosa, soundfile, yaml, transformers; "
        "assert sys.version_info[:2] >= (3, 10), 'Python 3.10+ is required'; "
        "print('SEED_VC_DEVICE=' + ('cuda' if torch.cuda.is_available() else 'cpu'), flush=True)"
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
        return False, "Checking the Seed-VC runtime timed out.", None
    except OSError:
        return False, "Could not start the configured Seed-VC Python environment.", None
    if result.returncode != 0:
        diagnostic = (result.stdout + "\n" + result.stderr).strip()
        for private_path in (str(ROOT), str(project), str(executable)):
            diagnostic = diagnostic.replace(private_path, "<private-path>")
        diagnostic = diagnostic[-1600:]
        suffix = f" Details: {diagnostic}" if diagnostic else ""
        return False, f"Seed-VC runtime/dependency check failed.{suffix}", None
    device_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith("SEED_VC_DEVICE=")),
        "SEED_VC_DEVICE=cpu",
    )
    device = device_line.removeprefix("SEED_VC_DEVICE=")
    if settings.seed_vc_device == "cuda" and device != "cuda":
        return False, "Seed-VC is configured for CUDA, but CUDA is unavailable in its isolated environment.", None
    return True, "Seed-VC zero-shot inference runtime and local DiT weights are ready.", device


def status() -> tuple[bool, str]:
    runtime_ok, detail, device = _runtime_status()
    if not runtime_ok:
        return False, detail
    selected = device if settings.seed_vc_device == "auto" else settings.seed_vc_device
    return True, (
        f"Seed-VC reference-conditioned voice conversion is ready on {selected} "
        f"(diffusion steps={settings.seed_vc_diffusion_steps}, fp16={settings.seed_vc_fp16})."
    )


def _capture_output(stream, tail: bytearray) -> None:
    while chunk := stream.read(4096):
        tail.extend(chunk)
        if len(tail) > MAX_LOG_BYTES:
            del tail[:-MAX_LOG_BYTES]


def _failure_message(log: str, returncode: int, hidden_paths: tuple[Path, ...] = ()) -> str:
    lowered = log.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return (
            "Seed-VC ran out of GPU memory. Try a shorter source clip, fewer diffusion steps "
            "(SEED_VC_DIFFUSION_STEPS=10), or set SEED_VC_DEVICE=cpu."
        )
    cleaned = log.replace(str(ROOT), "<project>").strip()
    for path in hidden_paths:
        cleaned = cleaned.replace(str(path), "<private-path>")
    # Drop absolute Windows paths that may still appear.
    cleaned = re.sub(r"[A-Za-z]:\\[^\s]+", "<private-path>", cleaned)
    return "Voice conversion failed: " + (cleaned[-2000:] or f"Seed-VC runner exited with code {returncode}.")


def convert(job, source: Path, reference: Path) -> dict[str, Path]:
    runtime_ok, detail, detected_device = _runtime_status()
    if not runtime_ok:
        raise AdapterError(f"Seed-VC is unavailable. {detail}")
    if settings.seed_vc_device == "cuda" and detected_device != "cuda":
        raise AdapterError("Seed-VC is configured for CUDA, but CUDA is unavailable in its isolated environment.")

    device = detected_device if settings.seed_vc_device == "auto" else settings.seed_vc_device
    output_path = storage.job_output_dir(job.job_id) / "converted.wav"
    work_dir = storage.job_output_dir(job.job_id) / "seed-vc-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(runtime_python()),
        str(ROOT / "backend" / "app" / "services" / "adapters" / "seed_vc_runner.py"),
        "--seed-vc-root", str(runtime_project()),
        "--source", str(source.resolve()),
        "--reference", str(reference.resolve()),
        "--output", str(output_path.resolve()),
        "--work-dir", str(work_dir.resolve()),
        "--checkpoint", str(checkpoint_path().resolve()),
        "--config", str(config_path().resolve()),
        "--diffusion-steps", str(settings.seed_vc_diffusion_steps),
        "--fp16", "true" if settings.seed_vc_fp16 and device == "cuda" else "false",
        "--device", device,
    ]
    job.message = "Converting voice with Seed-VC (reference-conditioned)"
    job.device = device

    # Inference is offline: model assets are installed only by the explicit setup command.
    try:
        process = subprocess.Popen(
            command,
            cwd=runtime_project(),
            env=_offline_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        shutil.rmtree(work_dir, ignore_errors=True)
        output_path.unlink(missing_ok=True)
        raise AdapterError("Could not start the Seed-VC runtime. Check SEED_VC_PYTHON and SEED_VC_PROJECT_DIR.")

    tail = bytearray()
    reader = threading.Thread(target=_capture_output, args=(process.stdout, tail), daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=settings.job_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join()
        shutil.rmtree(work_dir, ignore_errors=True)
        output_path.unlink(missing_ok=True)
        raise AdapterError(
            f"Voice conversion took longer than {settings.job_timeout_s} seconds and was stopped."
        )
    finally:
        if process.stdout:
            process.stdout.close()
    reader.join()
    output = bytes(tail).decode("utf-8", errors="replace")[-4000:]
    shutil.rmtree(work_dir, ignore_errors=True)

    if returncode != 0:
        output_path.unlink(missing_ok=True)
        raise AdapterError(_failure_message(
            output, returncode, (source, reference, runtime_project() or ROOT)
        ))
    try:
        audio_io.verify_wav(output_path, require_signal=True)
    except audio_io.AudioIOError as e:
        output_path.unlink(missing_ok=True)
        raise AdapterError(f"Voice conversion finished but the audio is not usable: {e}")
    return {"converted": output_path}
