"""Vocal separation with Demucs (HTDemucs).

Model repository : https://github.com/facebookresearch/demucs (archived by Meta; PyPI package `demucs` 4.1.0 is the one used)
Exact model      : `htdemucs` (Hybrid Transformer Demucs, 4 stems), safetensors weights from the
                   Hugging Face repo `adefossez/HTDemucs`. We merge drums+bass+other into "instrumental".
Licence          : code MIT. WEIGHTS: unresolved. The author has said the pretrained weights are not
                   covered by the code's MIT licence and are provided for scientific use. Treat the weights as
                   research/personal use only until you get written clarification. See docs/adapters/demucs.md.
Install          : PyTorch first (CUDA build), then `pip install -r requirements-ai.txt`.
Hardware         : NVIDIA GPU via CUDA, or CPU (much slower). Falls back to CPU if the GPU runs out of memory.
Input / output   : any FFmpeg-readable audio in; stereo 44.1 kHz 16-bit WAV out (vocals.wav, instrumental.wav).
Limitations      : bleed between vocals and instruments is normal; long files are slow; stems are re-encoded to
                   16-bit so they are not bit-exact with the input.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from importlib.util import find_spec
from pathlib import Path

from app.core.config import ROOT, settings
from app.services import audio_io, storage
from app.services.adapters import AdapterError

MODEL_NAME = "htdemucs"
HF_REPO = "adefossez/HTDemucs"
RUNNER_MODULE = "app.services.adapters.demucs_runner"
STEMS = ("vocals", "instrumental")


def hf_home() -> Path:
    return settings.models_dir / "hf"


def worker_env() -> dict:
    """Environment for the runner: keep weights inside ./models and forbid network access."""
    import os
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home())
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def weights_present() -> bool:
    snaps = hf_home() / "hub" / "models--adefossez--HTDemucs" / "snapshots"
    for yml in snaps.glob("*/htdemucs.yaml"):
        try:
            import yaml
            sigs = yaml.safe_load(yml.read_text(encoding="utf-8"))["models"]
        except Exception:
            return False
        return bool(sigs) and all((yml.parent / f"{s}.safetensors").exists() for s in sigs)
    return False


def status() -> tuple[bool, str]:
    if not (find_spec("demucs") and find_spec("torch")):
        return False, ("Not installed. With the backend venv active run: "
                       "pip install -r requirements-ai.txt (install PyTorch first).")
    if not shutil.which(settings.ffmpeg_path):
        return False, "FFmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg) and restart the backend."
    if not weights_present():
        return False, ("Model weights are not downloaded yet. From the project folder run: "
                       "python scripts/download_models.py demucs")
    return True, "htdemucs is ready. It uses the NVIDIA GPU when PyTorch can see one, otherwise the CPU."


def _clean(text: str) -> str:
    return text.replace(str(ROOT), "<project>").strip()[-500:]


def separate(job, src: Path) -> dict[str, Path]:
    """Run separation for `job`. Returns {"vocals": path, "instrumental": path} after verifying both files."""
    ok, detail = status()
    if not ok:
        raise AdapterError(f"Demucs is unavailable. {detail}")
    out_dir = storage.job_output_dir(job.job_id)
    cmd = [sys.executable, "-m", RUNNER_MODULE, str(src), str(out_dir), MODEL_NAME, settings.demucs_device]
    timed_out = threading.Event()
    result = None
    with tempfile.TemporaryFile("w+", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=settings.backend_dir, env=worker_env(), stdout=subprocess.PIPE,
                                stderr=err, text=True, encoding="utf-8")
        timer = threading.Timer(settings.job_timeout_s, lambda: (timed_out.set(), proc.kill()))
        timer.start()
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                job.message = msg.get("message", job.message)
                job.device = msg.get("device", job.device)
                if msg.get("stage") == "done":
                    result = msg
            proc.wait()
        finally:
            timer.cancel()
        if timed_out.is_set():
            raise AdapterError(f"Separation took longer than {settings.job_timeout_s} seconds and was stopped.")
        if proc.returncode != 0 or result is None:
            err.seek(0)
            raise AdapterError("Separation failed: " + (_clean(err.read()) or f"exit code {proc.returncode}"))
    outputs = {}
    for stem in STEMS:
        path = out_dir / f"{stem}.wav"
        try:
            audio_io.verify_wav(path)
        except audio_io.AudioIOError as e:
            raise AdapterError(f"Separation finished but the {stem} file is not usable: {e}")
        outputs[stem] = path
    return outputs
