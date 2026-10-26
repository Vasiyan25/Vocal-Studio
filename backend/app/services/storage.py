"""Private file storage. Files are addressed only by server-generated IDs."""
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from app.core.config import ROOT, settings

ID_RE = re.compile(r"^[a-f0-9]{32}$")
TEMP_PREFIX = "vocal-studio-"


def _temp_dir_for(name: str) -> Path:
    root = ROOT / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{TEMP_PREFIX}{name}-", dir=root))


def _ensure_temp_storage_path(current: Path, default_name: str) -> Path:
    legacy = ROOT / "storage" / default_name
    if current.exists() and current.is_dir() and str(current).startswith(str(ROOT / "storage")):
        if current.name.startswith(TEMP_PREFIX):
            return current
        # Migrate away from the long-lived legacy project storage path to a temp folder.
        temp_dir = _temp_dir_for(default_name)
        settings.__dict__["upload_dir" if default_name == "uploads" else "output_dir"] = temp_dir
        return temp_dir
    if current == legacy:
        temp_dir = _temp_dir_for(default_name)
        settings.__dict__["upload_dir" if default_name == "uploads" else "output_dir"] = temp_dir
        return temp_dir
    return current


def new_id() -> str:
    return uuid.uuid4().hex


def safe_filename(name: str) -> str:
    """Strip any path and unsafe characters from a client-supplied filename."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "audio")[:100]


def ensure_dirs() -> None:
    settings.upload_dir = _ensure_temp_storage_path(settings.upload_dir, "uploads")
    settings.output_dir = _ensure_temp_storage_path(settings.output_dir, "outputs")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)


def upload_path(file_id: str, ext: str) -> Path:
    return settings.upload_dir / f"{file_id}{ext}"


def find_upload(file_id: str) -> Path | None:
    if not ID_RE.match(file_id):
        return None
    for p in settings.upload_dir.glob(f"{file_id}.*"):
        return p
    return None


def job_output_dir(job_id: str) -> Path:
    """Private output folder for one job. Only reachable through the job endpoints."""
    if not ID_RE.match(job_id):
        raise ValueError("Invalid job id")
    path = settings.output_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_files() -> int:
    """Delete temporary uploads/outputs older than the configured TTL. Returns count removed."""
    cutoff = time.time() - settings.file_ttl_hours * 3600
    removed = 0
    for folder in (settings.upload_dir, settings.output_dir):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.name == ".gitkeep" or p.stat().st_mtime >= cutoff:
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
            elif p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
    legacy_root = ROOT / "storage"
    if legacy_root.exists():
        for path in legacy_root.iterdir():
            if path.is_dir() and path.name.startswith(TEMP_PREFIX):
                try:
                    if path.stat().st_mtime < cutoff:
                        shutil.rmtree(path, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
    return removed
