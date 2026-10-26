import json
import pytest
from fastapi.testclient import TestClient
import numpy as np

from app.core.config import settings
from app.main import app
from app.services import audio_io, storage
from app.services.adapters import demucs_adapter, kokoro_adapter, piper_adapter, rvc_adapter, seed_vc_adapter


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "up")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "max_upload_mb", 1)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def wav_bytes(tmp_path):
    p = tmp_path / "t.wav"
    samples = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)[None, :].astype("float32") * 0.5
    audio_io.write_wav(p, samples, 8000)
    return p.read_bytes()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_listed_and_unavailable_when_missing(client, monkeypatch):
    monkeypatch.setattr(demucs_adapter, "status", lambda: (False, "weights missing"))
    monkeypatch.setattr(piper_adapter, "status", lambda: (False, "voice missing"))
    monkeypatch.setattr(kokoro_adapter, "status", lambda: (False, "model missing"))
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (False, "runtime missing"))
    monkeypatch.setattr(rvc_adapter, "status", lambda: (False, "voice missing"))
    data = client.get("/api/models").json()
    assert {m["id"] for m in data} == {"demucs", "tts", "kokoro", "seed-vc", "rvc"}
    assert all(m["available"] is False for m in data)


def test_seed_vc_checkpoint_missing_is_unavailable_without_download_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")

    available, detail = seed_vc_adapter.status()

    assert available is False
    assert detail == "Required Seed-VC model files are missing."
    assert "checkpoint missing" not in detail.lower()
    assert "download it" not in detail.lower()


def test_kokoro_voice_listing_uses_installed_voice_packs(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    voice_dir = settings.models_dir / "kokoro" / "voices"
    voice_dir.mkdir(parents=True)
    (voice_dir / "af_heart.pt").write_bytes(b"installed voice pack")
    (voice_dir / "unknown_voice.pt").write_bytes(b"not a supported English voice id")

    voices = client.get("/api/tts/voices/kokoro").json()

    assert voices == [{"id": "af_heart", "name": "American English - Heart (female)"}]


def test_kokoro_model_status_requires_local_weights_and_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(kokoro_adapter, "find_spec", lambda _: object())

    available, _ = kokoro_adapter.status()
    assert not available

    model_dir = settings.models_dir / "kokoro"
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"vocab": {}, "n_token": 1, "plbert": {}}),
        encoding="utf-8",
    )
    (model_dir / "kokoro-v1_0.pth").write_bytes(b"weights")
    (model_dir / "voices" / "af_heart.pt").write_bytes(b"voice")

    available, _ = kokoro_adapter.status()
    assert available


def test_upload_valid_wav_and_stream(client, wav_bytes):
    r = client.post("/api/audio/upload", files={"file": ("../../evil name.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["file_id"]) == 32 and "/" not in body["filename"]
    if body["duration_seconds"] is not None:
        assert abs(body["duration_seconds"] - 1) < 0.1
    s = client.get(f"/api/audio/{body['file_id']}/stream")
    assert s.status_code == 200 and s.content == wav_bytes


def test_upload_rejects_bad_extension(client):
    r = client.post("/api/audio/upload", files={"file": ("a.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_rejects_fake_audio(client):
    r = client.post("/api/audio/upload", files={"file": ("fake.wav", b"this is not audio at all", "audio/wav")})
    assert r.status_code == 400


def test_upload_rejects_too_large(client, wav_bytes):
    big = wav_bytes + b"\0" * (2 * 1024 * 1024)
    r = client.post("/api/audio/upload", files={"file": ("big.wav", big, "audio/wav")})
    assert r.status_code == 413


def test_separate_unavailable_returns_503(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(demucs_adapter, "status", lambda: (False, "weights missing"))
    fid = client.post("/api/audio/upload", files={"file": ("a.wav", wav_bytes, "audio/wav")}).json()["file_id"]
    r = client.post("/api/audio/separate", json={"file_id": fid})
    assert r.status_code == 503 and "Demucs" in r.json()["detail"]


def test_tts_and_convert_unavailable(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(piper_adapter, "status", lambda: (False, "voice missing"))
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (False, "Required Seed-VC model files are missing."))
    monkeypatch.setattr(rvc_adapter, "status", lambda: (False, "voice missing"))
    assert client.post("/api/audio/tts", json={"text": "hello"}).status_code == 503
    fid = client.post("/api/audio/upload", files={"file": ("a.wav", wav_bytes, "audio/wav")}).json()["file_id"]
    ref = client.post("/api/audio/upload", files={"file": ("b.wav", wav_bytes, "audio/wav")}).json()["file_id"]
    response = client.post("/api/audio/convert", json={"file_id": fid, "reference_file_id": ref})
    assert response.status_code == 503
    assert response.json()["detail"] == "Seed-VC is unavailable. Required Seed-VC model files are missing."
    assert client.post("/api/audio/convert-rvc", json={"file_id": fid, "voice_model": "x"}).status_code == 503


def test_voices_endpoint_lists_installed_models(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    voice_dir = settings.models_dir / "rvc"
    voice_dir.mkdir(parents=True)
    (voice_dir / "authorized-voice.pth").write_bytes(b"model")
    (voice_dir / "empty.pth").write_bytes(b"")
    (voice_dir / "invalid name.pth").write_bytes(b"model")
    assert client.get("/api/voices").json() == ["authorized-voice"]


def test_convert_rejects_invalid_ids_and_missing_upload(client, monkeypatch):
    monkeypatch.setattr(seed_vc_adapter, "status", lambda: (True, "ok"))
    invalid = client.post(
        "/api/audio/convert",
        json={"file_id": "a" * 32, "reference_file_id": "not-hex"},
    )
    assert invalid.status_code == 422
    missing_upload = client.post(
        "/api/audio/convert",
        json={"file_id": "a" * 32, "reference_file_id": "b" * 32},
    )
    assert missing_upload.status_code == 404


def test_unknown_job_and_bad_ids(client):
    assert client.get("/api/jobs/abc").status_code == 404
    assert client.get("/api/jobs/abc/download").status_code == 404
    assert client.get("/api/audio/../../etc/passwd/stream").status_code == 404
    assert client.post("/api/audio/separate", json={"file_id": "../x"}).status_code == 422


def test_completed_job_is_recovered_from_output_metadata(client, wav_bytes):
    job_id = storage.new_id()
    output_dir = settings.output_dir / job_id
    output_dir.mkdir(parents=True)
    (output_dir / "speech.wav").write_bytes(wav_bytes)
    (settings.output_dir / f".job-{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "kind": "tts", "status": "completed", "message": "Done", "outputs": ["speech"],
    }), encoding="utf-8")

    status = client.get(f"/api/jobs/{job_id}")
    result = client.get(f"/api/jobs/{job_id}/download?stem=speech")

    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["outputs"] == ["speech"]
    assert result.status_code == 200 and result.content == wav_bytes


def test_interrupted_job_is_recovered_as_failed(client):
    job_id = storage.new_id()
    output_dir = settings.output_dir / job_id
    output_dir.mkdir(parents=True)
    (settings.output_dir / f".job-{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "kind": "convert", "status": "running", "message": "Converting", "outputs": [],
    }), encoding="utf-8")

    response = client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "backend restarted" in response.json()["error"].lower()
