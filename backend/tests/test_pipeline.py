"""Tests for the job pipeline and adapter process handling.

IMPORTANT: these use a fake runner / fake adapter functions ONLY to test the plumbing (jobs, polling,
download, errors, timeouts, path safety). They prove nothing about Demucs or Piper audio quality.
Real models are verified by running the app on a machine that has them (see docs/milestone-2-3.md).
"""
import time
import shutil
import io
import os
import sys
import types
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import audio_io, storage
from app.services import audio_validation
from app.services.adapters import AdapterError, demucs_adapter, kokoro_adapter, piper_adapter, rvc_adapter, seed_vc_adapter
from app.services.adapters import rvc_runner


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "up")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def tone(seconds=1.0, sr=8000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return np.sin(2 * np.pi * 330 * t)[None, :].astype("float32") * 0.5


@pytest.fixture
def uploaded(client, tmp_path):
    p = tmp_path / "in.wav"
    audio_io.write_wav(p, tone(), 8000)
    r = client.post("/api/audio/upload", files={"file": ("in.wav", p.read_bytes(), "audio/wav")})
    return r.json()["file_id"]


def wait(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def fake_separate(job, src):
    out = settings.output_dir / job.job_id
    out.mkdir(parents=True)
    audio_io.write_wav(out / "vocals.wav", tone(), 8000)
    audio_io.write_wav(out / "instrumental.wav", tone(), 8000)
    job.device = "cpu"
    return {"vocals": out / "vocals.wav", "instrumental": out / "instrumental.wav"}


def test_audio_io_roundtrip_and_verify(tmp_path):
    p = tmp_path / "a.wav"
    audio_io.write_wav(p, np.concatenate([tone(), tone() * 5], axis=1), 8000)  # >1.0 peak gets rescaled, not clipped hard
    assert abs(audio_io.verify_wav(p, require_signal=True) - 2.0) < 0.01
    if not shutil.which(settings.ffmpeg_path):
        pytest.skip("FFmpeg is required to test audio decoding.")
    back = audio_io.decode(p, 8000, 1)
    assert back.shape[0] == 1 and np.abs(back).max() <= 1.0
    with pytest.raises(audio_io.AudioIOError):
        audio_io.verify_wav(tmp_path / "missing.wav")
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    with pytest.raises(audio_io.AudioIOError):
        audio_io.verify_wav(tmp_path / "bad.wav")
    silent = tmp_path / "s.wav"
    audio_io.write_wav(silent, np.zeros((1, 800), dtype="float32"), 8000)
    with pytest.raises(audio_io.AudioIOError):
        audio_io.verify_wav(silent, require_signal=True)


def test_separate_job_flow_stream_and_download(client, uploaded, monkeypatch):
    monkeypatch.setattr(demucs_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(demucs_adapter, "separate", fake_separate)
    r = client.post("/api/audio/separate", json={"file_id": uploaded})
    assert r.status_code == 202
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "completed" and job["outputs"] == ["vocals", "instrumental"] and job["device"] == "cpu"
    jid = job["job_id"]
    for stem in ("vocals", "instrumental"):
        d = client.get(f"/api/jobs/{jid}/download", params={"stem": stem})
        assert d.status_code == 200 and d.content[:4] == b"RIFF"
        assert "attachment" in d.headers["content-disposition"]
        s = client.get(f"/api/jobs/{jid}/stream", params={"stem": stem})
        assert s.status_code == 200 and "attachment" not in s.headers.get("content-disposition", "")
    assert client.get(f"/api/jobs/{jid}/download").status_code == 200  # default stem
    assert client.get(f"/api/jobs/{jid}/download", params={"stem": "../../etc/passwd"}).status_code == 404


def test_failed_job_reports_error_and_has_no_download(client, uploaded, monkeypatch):
    def boom(job, src):
        raise AdapterError("model exploded")
    monkeypatch.setattr(demucs_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(demucs_adapter, "separate", boom)
    job = wait(client, client.post("/api/audio/separate", json={"file_id": uploaded}).json()["job_id"])
    assert job["status"] == "failed" and job["error"] == "model exploded"
    assert client.get(f"/api/jobs/{job['job_id']}/download").status_code == 409


def test_unexpected_crash_does_not_leak_details(client, uploaded, monkeypatch):
    def boom(job, src):
        raise RuntimeError("secret C:\\path")
    monkeypatch.setattr(demucs_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(demucs_adapter, "separate", boom)
    job = wait(client, client.post("/api/audio/separate", json={"file_id": uploaded}).json()["job_id"])
    assert job["status"] == "failed" and "secret" not in job["error"]


def test_tts_job_flow(client, monkeypatch):
    def fake_tts(job, text, voice_id=None):
        out = settings.output_dir / job.job_id
        out.mkdir(parents=True)
        audio_io.write_wav(out / "speech.wav", tone(), 8000)
        return {"speech": out / "speech.wav"}
    monkeypatch.setattr(piper_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(piper_adapter, "synthesize", fake_tts)
    assert client.post("/api/audio/tts", json={"text": ""}).status_code == 422
    assert client.post("/api/audio/tts", json={"text": "   "}).status_code == 422
    r = client.post("/api/audio/tts", json={"text": "hello world"})
    assert r.status_code == 202
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "completed" and job["outputs"] == ["speech"]
    assert client.get(f"/api/jobs/{job['job_id']}/download").content[:4] == b"RIFF"


def test_kokoro_tts_job_flow_uses_selected_installed_voice(client, monkeypatch):
    voice = kokoro_adapter.Voice(
        id="af_heart",
        name="American English - Heart (female)",
        path=Path("af_heart.pt"),
    )

    def fake_synthesize(job, text, voice_id):
        assert text == "Keep this text"
        assert voice_id == "af_heart"
        out = settings.output_dir / job.job_id
        out.mkdir(parents=True)
        audio_io.write_wav(out / "speech.wav", tone(), 8000)
        return {"speech": out / "speech.wav"}

    monkeypatch.setattr(kokoro_adapter, "status", lambda: (True, "ready"))
    monkeypatch.setattr(kokoro_adapter, "list_voices", lambda: [voice])
    monkeypatch.setattr(kokoro_adapter, "synthesize", fake_synthesize)
    response = client.post("/api/audio/tts", json={
        "text": "Keep this text",
        "voice_id": "af_heart",
        "model_id": "kokoro",
    })

    assert response.status_code == 202
    job = wait(client, response.json()["job_id"])
    assert job["status"] == "completed" and job["outputs"] == ["speech"]
    assert client.get(f"/api/jobs/{job['job_id']}/stream").content[:4] == b"RIFF"
    assert client.get(f"/api/jobs/{job['job_id']}/download").content[:4] == b"RIFF"


def test_kokoro_unavailable_returns_error_without_queueing(client, monkeypatch):
    monkeypatch.setattr(kokoro_adapter, "status", lambda: (False, "Model files are missing."))
    response = client.post("/api/audio/tts", json={"text": "hello", "model_id": "kokoro"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Kokoro is unavailable. Model files are missing."


def test_kokoro_rejects_unknown_voice(client, monkeypatch):
    monkeypatch.setattr(kokoro_adapter, "status", lambda: (True, "ready"))
    monkeypatch.setattr(kokoro_adapter, "list_voices", lambda: [])
    response = client.post("/api/audio/tts", json={"text": "hello", "model_id": "kokoro", "voice_id": "missing"})
    assert response.status_code == 404


def test_tts_respects_configured_character_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "max_tts_characters", 3)
    response = client.post("/api/audio/tts", json={"text": "four"})
    assert response.status_code == 422
    assert "3 characters" in response.json()["detail"]


def test_convert_job_flow(client, uploaded, monkeypatch, tmp_path):
    def fake_convert(job, source, reference):
        out = settings.output_dir / job.job_id
        out.mkdir(parents=True)
        audio_io.write_wav(out / "converted.wav", tone(), 8000)
        job.device = "cpu"
        return {"converted": out / "converted.wav"}

    ref_path = tmp_path / "ref.wav"
    audio_io.write_wav(ref_path, tone(0.5), 8000)
    ref = client.post("/api/audio/upload", files={"file": ("ref.wav", ref_path.read_bytes(), "audio/wav")}).json()

    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(seed_vc_adapter, "convert", fake_convert)
    response = client.post("/api/audio/convert", json={
        "file_id": uploaded,
        "reference_file_id": ref["file_id"],
    })
    assert response.status_code == 202
    job = wait(client, response.json()["job_id"])
    assert job["status"] == "completed" and job["outputs"] == ["converted"]
    assert client.get(f"/api/jobs/{job['job_id']}/download").content[:4] == b"RIFF"


def test_convert_rejects_same_source_and_reference(client, uploaded, monkeypatch):
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ok"))
    response = client.post("/api/audio/convert", json={
        "file_id": uploaded,
        "reference_file_id": uploaded,
    })
    assert response.status_code == 400


def test_convert_rejects_missing_reference(client, uploaded, monkeypatch):
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ok"))
    response = client.post("/api/audio/convert", json={
        "file_id": uploaded,
        "reference_file_id": "a" * 32,
    })
    assert response.status_code == 404


def test_convert_rejects_invalid_source_and_reference_audio(client, monkeypatch, tmp_path):
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ok"))
    source_id = "c" * 32
    reference_id = "d" * 32
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    storage.upload_path(source_id, ".wav").write_bytes(b"not a wav")
    storage.upload_path(reference_id, ".wav").write_bytes(b"not a wav")
    response = client.post("/api/audio/convert", json={
        "file_id": source_id,
        "reference_file_id": reference_id,
    })
    assert response.status_code == 400
    assert "do not look like" in response.json()["detail"]


def test_seed_vc_failed_job_preserves_real_adapter_error(client, uploaded, monkeypatch, tmp_path):
    reference_path = tmp_path / "reference.wav"
    audio_io.write_wav(reference_path, tone(0.5), 8000)
    reference_id = client.post(
        "/api/audio/upload",
        files={"file": ("reference.wav", reference_path.read_bytes(), "audio/wav")},
    ).json()["file_id"]

    def fail_conversion(job, source, reference):
        raise AdapterError("Seed-VC inference failed because required model weights could not be loaded.")

    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ready"))
    monkeypatch.setattr(seed_vc_adapter, "convert", fail_conversion)
    response = client.post("/api/audio/convert", json={
        "file_id": uploaded,
        "reference_file_id": reference_id,
    })
    job = wait(client, response.json()["job_id"])

    assert response.status_code == 202
    assert job["status"] == "failed"
    assert "weights could not be loaded" in job["error"]
    assert client.get(f"/api/jobs/{job['job_id']}/download").status_code == 409


def test_convert_rvc_job_flow(client, uploaded, monkeypatch):
    def fake_convert(job, source, voice_model):
        out = settings.output_dir / job.job_id
        out.mkdir(parents=True)
        audio_io.write_wav(out / "converted.wav", tone(), 8000)
        job.device = "cpu"
        return {"converted": out / "converted.wav"}

    monkeypatch.setattr(rvc_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(rvc_adapter, "find_voice", lambda voice_id: object() if voice_id == "authorized" else None)
    monkeypatch.setattr(rvc_adapter, "convert", fake_convert)
    response = client.post("/api/audio/convert-rvc", json={"file_id": uploaded, "voice_model": "authorized"})
    assert response.status_code == 202
    job = wait(client, response.json()["job_id"])
    assert job["status"] == "completed" and job["outputs"] == ["converted"]


def test_convert_rvc_rejects_unknown_voice(client, uploaded, monkeypatch):
    monkeypatch.setattr(rvc_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(rvc_adapter, "find_voice", lambda voice_id: None)
    response = client.post("/api/audio/convert-rvc", json={"file_id": uploaded, "voice_model": "missing"})
    assert response.status_code == 404


def test_seed_vc_adapter_missing_runtime(monkeypatch):
    monkeypatch.setattr(settings, "seed_vc_python", None)
    monkeypatch.setattr(settings, "seed_vc_project_dir", None)
    ok, detail = seed_vc_adapter.status()
    assert not ok and "SEED_VC_PYTHON" in detail


def test_seed_vc_adapter_uses_source_and_reference(tmp_path, monkeypatch):
    project = tmp_path / "seed-vc-project"
    project.mkdir()
    (project / "inference.py").write_text("# stub", encoding="utf-8")
    weights = settings.models_dir / "seed-vc"
    weights.mkdir(parents=True)
    (weights / seed_vc_adapter.DIT_CHECKPOINT_NAME).write_bytes(b"ckpt")
    (weights / seed_vc_adapter.DIT_CONFIG_NAME).write_text("model: {}", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"x")
    monkeypatch.setattr(settings, "seed_vc_python", python)
    monkeypatch.setattr(settings, "seed_vc_project_dir", project)
    monkeypatch.setattr(seed_vc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output = Path(command[command.index("--output") + 1])
        audio_io.write_wav(output, tone(), 8000)
        return SimpleNamespace(stdout=io.BytesIO(), wait=lambda timeout=None: 0)

    monkeypatch.setattr(seed_vc_adapter.subprocess, "Popen", fake_run)
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    audio_io.write_wav(source, tone(), 8000)
    audio_io.write_wav(reference, tone(0.4), 8000)
    job = SimpleNamespace(job_id="e" * 32, message="", device=None)
    outputs = seed_vc_adapter.convert(job, source, reference)
    assert outputs["converted"].exists()
    assert captured["command"][captured["command"].index("--source") + 1] == str(source.resolve())
    assert captured["command"][captured["command"].index("--reference") + 1] == str(reference.resolve())


def test_rvc_adapter_uses_selected_model_and_index(tmp_path, monkeypatch):
    model_dir = settings.models_dir / "rvc"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "authorized.pth"
    index_path = model_dir / "authorized.index"
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    monkeypatch.setattr(rvc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    captured = {}

    monkeypatch.setattr(settings, "rvc_python", tmp_path / "rvc-python.exe")
    monkeypatch.setattr(settings, "rvc_project_dir", tmp_path / "rvc-project")
    settings.rvc_project_dir.mkdir()
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        audio_io.write_wav(output, tone(), 8000)
        return SimpleNamespace(
            stdout=io.BytesIO(),
            wait=lambda timeout=None: 0,
        )

    monkeypatch.setattr(rvc_adapter.subprocess, "Popen", fake_run)
    job = SimpleNamespace(job_id="a" * 32, message="", device=None)
    outputs = rvc_adapter.convert(job, tmp_path / "input.wav", "authorized")
    assert outputs["converted"].exists()
    assert captured["command"][captured["command"].index("--voice-id") + 1] == "authorized"
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_rvc_adapter_missing_runtime_and_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "rvc_python", None)
    monkeypatch.setattr(settings, "rvc_project_dir", None)
    ok, detail = rvc_adapter.status()
    assert not ok and "RVC_PYTHON" in detail

    monkeypatch.setattr(rvc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    ok, detail = rvc_adapter.status()
    assert not ok and "No non-empty voice checkpoints" in detail


def test_rvc_adapter_discovers_only_safe_nonempty_checkpoints(monkeypatch, tmp_path):
    model_dir = tmp_path / "models" / "rvc"
    model_dir.mkdir(parents=True)
    (model_dir / "good.pth").write_bytes(b"checkpoint")
    (model_dir / "bad name.pth").write_bytes(b"checkpoint")
    (model_dir / "empty.pth").write_bytes(b"")
    (model_dir / "linked.pth").write_bytes(b"checkpoint")
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink",
        lambda path: path.name == "linked.pth" or original_is_symlink(path),
    )
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    assert [voice.id for voice in rvc_adapter.list_voices()] == ["good"]


def test_rvc_runner_calls_official_cli_offline_with_optional_index(monkeypatch, tmp_path):
    project = tmp_path / "rvc-project"
    (project / "infer").mkdir(parents=True)
    (project / "infer" / "cli.py").write_text("", encoding="utf-8")
    models = tmp_path / "models" / "rvc"
    models.mkdir(parents=True)
    (models / "default.pth").write_bytes(b"checkpoint")
    (models / "default.index").write_bytes(b"index")
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    output = tmp_path / "output.wav"
    calls = []
    fake_cli = types.ModuleType("infer.cli")
    fake_cli.load_model_metadata = lambda model_path: (2, [{"id": 3, "name": "Speaker 3"}])
    fake_cli.select_speaker = lambda count, speakers, requested: (
        requested if requested is not None else speakers[0]["id"]
    )

    def fake_main(args):
        calls.append(args)
        audio_io.write_wav(output, tone(), 8000)
        return 0
    fake_cli.main = fake_main
    fake_infer = types.ModuleType("infer")
    fake_infer.__path__ = []
    monkeypatch.setitem(sys.modules, "infer", fake_infer)
    monkeypatch.setitem(sys.modules, "infer.cli", fake_cli)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(sys, "argv", [
        "rvc_runner.py",
        "--rvc-root", str(project),
        "--models-dir", str(models),
        "--voice-id", "default",
        "--input", str(source),
        "--output", str(output),
        "--device", "cpu",
    ])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(tmp_path)
    rvc_runner.main()
    args = calls[0]
    assert args[args.index("--model") + 1] == str(models / "default.pth")
    assert args[args.index("--speaker-id") + 1] == "3"
    assert args[args.index("--index") + 1] == str(models / "default.index")
    assert args[args.index("--index-rate") + 1] == "0.75"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONIOENCODING"] == "utf-8"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert args[args.index("--input") + 1] == str(source)
    assert args[args.index("--output") + 1] == str(output)
    assert output.is_file() and output.stat().st_size > 0
    (models / "default.index").unlink()
    rvc_runner.main()
    args = calls[1]
    assert "--index" not in args
    assert args[args.index("--index-rate") + 1] == "0"


def test_rvc_runner_rejects_missing_model_and_missing_output(monkeypatch, tmp_path):
    project = tmp_path / "rvc-project"
    (project / "infer").mkdir(parents=True)
    (project / "infer" / "cli.py").write_text("", encoding="utf-8")
    models = tmp_path / "models" / "rvc"
    models.mkdir(parents=True)
    source = tmp_path / "input.wav"
    audio_io.write_wav(source, tone(), 8000)
    output = tmp_path / "output.wav"
    fake_cli = types.ModuleType("infer.cli")
    fake_cli.main = lambda args: 0
    fake_infer = types.ModuleType("infer")
    fake_infer.__path__ = []
    monkeypatch.setitem(sys.modules, "infer", fake_infer)
    monkeypatch.setitem(sys.modules, "infer.cli", fake_cli)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", [
        "rvc_runner.py", "--rvc-root", str(project), "--models-dir", str(models),
        "--voice-id", "default", "--input", str(source), "--output", str(output),
        "--device", "cpu",
    ])
    fake_cli.load_model_metadata = lambda model_path: (1, [])
    fake_cli.select_speaker = lambda count, speakers, requested: 0 if requested is None else requested
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        rvc_runner.main()

    (models / "default.pth").write_bytes(b"trusted test fixture")
    fake_cli.main = lambda args: 7
    with pytest.raises(RuntimeError, match="exit status 7"):
        rvc_runner.main()

    fake_cli.main = lambda args: 0
    with pytest.raises(RuntimeError, match="without generating"):
        rvc_runner.main()


def test_rvc_runtime_failure_reports_dependency_and_import_diagnostics(monkeypatch, tmp_path):
    project = tmp_path / "rvc-project"
    (project / "infer").mkdir(parents=True)
    (project / "infer" / "cli.py").write_text("", encoding="utf-8")
    hubert = project / "assets" / "hubert_base"
    hubert.mkdir(parents=True)
    (hubert / "config.json").write_text("{}", encoding="utf-8")
    (hubert / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (hubert / "pytorch_model.bin").write_bytes(b"test")
    rmvpe = project / "assets" / "rmvpe"
    rmvpe.mkdir(parents=True)
    (rmvpe / "rmvpe.pt").write_bytes(b"test")
    executable = tmp_path / "rvc-python.exe"
    executable.write_bytes(b"test")
    monkeypatch.setattr(settings, "rvc_python", executable)
    monkeypatch.setattr(settings, "rvc_project_dir", project)
    monkeypatch.setattr(rvc_adapter.shutil, "which", lambda _: "ffmpeg.exe")
    captured = {}

    def failed_check(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=1,
            stdout="RVC_DEPENDENCIES=numba=0.68.0; llvmlite=0.50.0; numpy=1.26.4",
            stderr="ImportError: DLL load failed while importing _devicearray: blocked by policy",
        )

    monkeypatch.setattr(rvc_adapter.subprocess, "run", failed_check)
    ok, detail, device = rvc_adapter._runtime_status()
    assert not ok and device is None
    assert "numba=0.68.0" in detail and "llvmlite=0.50.0" in detail
    assert "numpy=1.26.4" in detail and "_devicearray" in detail
    assert "numba._devicearray" in captured["command"][2]
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_rvc_adapter_timeout_cleans_partial_output(monkeypatch, tmp_path):
    model_dir = settings.models_dir / "rvc"
    model_dir.mkdir(parents=True)
    (model_dir / "authorized.pth").write_bytes(b"model")
    monkeypatch.setattr(rvc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    monkeypatch.setattr(settings, "job_timeout_s", 1)
    monkeypatch.setattr(settings, "rvc_python", tmp_path / "rvc-python.exe")
    monkeypatch.setattr(settings, "rvc_project_dir", tmp_path / "rvc-project")
    settings.rvc_project_dir.mkdir()

    class TimedOut:
        def __init__(self, command, **kwargs):
            self.output = Path(command[command.index("--output") + 1])
            audio_io.write_wav(self.output, tone(), 8000)
            self.killed = False
            self.stdout = io.BytesIO()

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("rvc", timeout)
            return -9

        def kill(self):
            self.killed = True

    monkeypatch.setattr(rvc_adapter.subprocess, "Popen", TimedOut)
    job = SimpleNamespace(job_id="b" * 32, message="", device=None)
    with pytest.raises(AdapterError, match="took longer"):
        rvc_adapter.convert(job, tmp_path / "input.wav", "authorized")
    assert not (settings.output_dir / job.job_id / "converted.wav").exists()


@pytest.mark.parametrize(
    ("returncode", "log_text", "expected"),
    [
        (1, "invalid model format", "Voice conversion failed"),
        (1, "CUDA out of memory", "ran out of GPU memory"),
        (1, "RVC 模型加载失败", "RVC 模型加载失败"),
    ],
)
def test_rvc_adapter_failed_subprocess_cleans_output(monkeypatch, tmp_path, returncode, log_text, expected):
    model_dir = settings.models_dir / "rvc"
    model_dir.mkdir(parents=True)
    (model_dir / "authorized.pth").write_bytes(b"model")
    monkeypatch.setattr(rvc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    monkeypatch.setattr(settings, "rvc_python", tmp_path / "rvc-python.exe")
    monkeypatch.setattr(settings, "rvc_project_dir", tmp_path / "rvc-project")
    settings.rvc_project_dir.mkdir()

    class Failed:
        def __init__(self, command, **kwargs):
            self.output = Path(command[command.index("--output") + 1])
            audio_io.write_wav(self.output, tone(), 8000)
            self.stdout = io.BytesIO(log_text.encode())

        def wait(self, timeout=None):
            return returncode

    monkeypatch.setattr(rvc_adapter.subprocess, "Popen", Failed)
    job = SimpleNamespace(job_id="c" * 32, message="", device=None)
    with pytest.raises(AdapterError, match=expected):
        rvc_adapter.convert(job, tmp_path / "input.wav", "authorized")
    assert not (settings.output_dir / job.job_id / "converted.wav").exists()


def test_rvc_adapter_rejects_malformed_success_output(monkeypatch, tmp_path):
    model_dir = settings.models_dir / "rvc"
    model_dir.mkdir(parents=True)
    (model_dir / "authorized.pth").write_bytes(b"model")
    monkeypatch.setattr(rvc_adapter, "_runtime_status", lambda: (True, "ready", "cpu"))
    monkeypatch.setattr(settings, "rvc_python", tmp_path / "rvc-python.exe")
    monkeypatch.setattr(settings, "rvc_project_dir", tmp_path / "rvc-project")
    settings.rvc_project_dir.mkdir()

    class EmptySuccess:
        def __init__(self, command, **kwargs):
            self.output = Path(command[command.index("--output") + 1])
            self.output.write_bytes(b"not a wav")
            self.stdout = io.BytesIO()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(rvc_adapter.subprocess, "Popen", EmptySuccess)
    job = SimpleNamespace(job_id="d" * 32, message="", device=None)
    with pytest.raises(AdapterError, match="not usable"):
        rvc_adapter.convert(job, tmp_path / "input.wav", "authorized")
    assert not (settings.output_dir / job.job_id / "converted.wav").exists()


def test_separate_rejects_overlong_audio(client, uploaded, monkeypatch):
    monkeypatch.setattr(demucs_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(settings, "max_audio_minutes", 0)
    monkeypatch.setattr(
        audio_validation, "verify_audio",
        lambda *args, **kwargs: SimpleNamespace(duration=1.0),
    )
    assert client.post("/api/audio/separate", json={"file_id": uploaded}).status_code == 413


# ---- adapter process handling, using tests/fake_runner.py instead of Demucs ----

@pytest.fixture
def fake_runner(monkeypatch):
    monkeypatch.setattr(demucs_adapter, "status", lambda: (True, "ok"))
    monkeypatch.setattr(demucs_adapter, "RUNNER_MODULE", "tests.fake_runner")


class J:
    job_id = "a" * 32
    message = ""
    device = None


def test_adapter_success_parses_progress_and_verifies_files(fake_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ok")
    j = J()
    out = demucs_adapter.separate(j, tmp_path / "x.wav")
    assert set(out) == {"vocals", "instrumental"} and all(p.exists() for p in out.values())
    assert j.message == "Fake separating" and j.device == "cpu"


def test_adapter_crash_gives_cleaned_error(fake_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    with pytest.raises(AdapterError, match="Separation failed"):
        demucs_adapter.separate(J(), tmp_path / "x.wav")


def test_adapter_missing_output_file_is_caught(fake_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "missing_instrumental")
    with pytest.raises(AdapterError, match="instrumental"):
        demucs_adapter.separate(J(), tmp_path / "x.wav")


def test_adapter_timeout(fake_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setattr(settings, "job_timeout_s", 1)
    with pytest.raises(AdapterError, match="longer than"):
        demucs_adapter.separate(J(), tmp_path / "x.wav")


def test_status_messages_explain_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(demucs_adapter, "find_spec", lambda name: None)
    ok, detail = demucs_adapter.status()
    assert not ok and "requirements-ai.txt" in detail
    monkeypatch.setattr(piper_adapter, "find_spec", lambda name: object())
    ok, detail = piper_adapter.status()
    assert not ok and "download_models.py piper" in detail
