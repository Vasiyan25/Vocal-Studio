"""Job metadata store and a single background worker. Replaced by a database in Milestone 5.

One worker runs AI jobs one at a time, so two heavy jobs never fight over the 6 GB GPU and the API
stays responsive while a job runs. Small job records live beside their outputs so completed work
can be recovered after a backend restart; interrupted work is recorded as failed.
"""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.services.adapters import AdapterError
from app.core.config import settings
from app.services import storage
from app.services.storage import new_id

log = logging.getLogger("vocal-studio.jobs")
JOB_TYPES = {"separate", "tts", "convert"}
JOB_STATUSES = {"queued", "running", "completed", "failed"}
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "queued"
    message: str = "Waiting to start"
    device: Optional[str] = None
    error: Optional[str] = None
    outputs: dict[str, Path] = field(default_factory=dict)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_persist_lock = threading.Lock()
_loaded_dirs: set[Path] = set()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-job")


def _persist_job(job: Job) -> None:
    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = settings.output_dir / f".job-{job.job_id}.json"
        payload = {
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "message": job.message,
            "device": job.device,
            "error": job.error,
            "outputs": list(job.outputs),
        }
        with _persist_lock:
            temporary = metadata.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(metadata)
    except OSError:
        log.exception("Could not persist metadata for job %s", job.job_id)


def _load_persisted_jobs_locked() -> None:
    root = settings.output_dir.resolve()
    if root in _loaded_dirs:
        return
    _loaded_dirs.add(root)
    if not root.is_dir():
        return

    for metadata in root.glob(".job-*.json"):
        job_id = metadata.name[len(".job-"):-len(".json")]
        if not storage.ID_RE.fullmatch(job_id):
            continue
        folder = root / job_id
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if data.get("job_id") != job_id or data.get("kind") not in JOB_TYPES:
                raise ValueError("invalid job identity")
            status = data.get("status")
            if status not in JOB_STATUSES:
                raise ValueError("invalid job status")

            outputs = {}
            for name in data.get("outputs", []):
                if not isinstance(name, str) or not OUTPUT_NAME_RE.fullmatch(name):
                    raise ValueError("invalid output name")
                output_path = folder / f"{name}.wav"
                if output_path.is_file():
                    outputs[name] = output_path

            if status in {"queued", "running"}:
                status = "failed"
                message = "Interrupted by a backend restart"
                error = "The backend restarted while this task was running. Start the task again."
                outputs = {}
            elif data["status"] == "completed" and not outputs:
                status = "failed"
                message = "Task output is unavailable"
                error = "The completed task's output files are missing or have expired. Run the task again."
            else:
                message = str(data.get("message") or "")
                error = data.get("error")

            job = Job(
                job_id=job_id,
                kind=data["kind"],
                status=status,
                message=message,
                device=data.get("device") if isinstance(data.get("device"), str) else None,
                error=error if isinstance(error, str) else None,
                outputs=outputs,
            )
            _jobs[job.job_id] = job
            if data["status"] != job.status:
                _persist_job(job)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Skipping unreadable job metadata at %s: %s", metadata, exc)


def get_job(job_id: str) -> Job | None:
    with _lock:
        _load_persisted_jobs_locked()
        return _jobs.get(job_id)


def submit(kind: str, fn: Callable[..., dict[str, Path]], *args) -> Job:
    """Queue fn(job, *args). fn returns {output_name: Path}."""
    job = Job(job_id=new_id(), kind=kind)
    with _lock:
        _load_persisted_jobs_locked()
        _jobs[job.job_id] = job
    _persist_job(job)
    _pool.submit(_run, job, fn, args)
    return job


def _run(job: Job, fn, args) -> None:
    with _lock:
        job.status, job.message = "running", "Starting"
    _persist_job(job)
    try:
        outputs = fn(job, *args)
        with _lock:
            job.outputs = outputs
            job.status, job.message = "completed", "Done"
    except AdapterError as e:
        with _lock:
            job.status, job.error, job.message = "failed", str(e), "Failed"
    except Exception:
        log.exception("Job %s crashed", job.job_id)
        with _lock:
            job.status, job.error, job.message = "failed", "Unexpected error. Check the backend terminal for details.", "Failed"
    _persist_job(job)
