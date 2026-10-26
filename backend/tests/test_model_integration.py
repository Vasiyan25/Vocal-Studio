"""Opt-in real-model smoke tests; these are skipped until locally installed assets are available."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.services import audio_io
from app.services.adapters import kokoro_adapter, seed_vc_adapter


def test_real_kokoro_generation_when_installed(tmp_path, monkeypatch):
    available, detail = kokoro_adapter.status()
    if not available:
        pytest.skip(detail)

    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    voice = kokoro_adapter.list_voices()[0]
    job = SimpleNamespace(job_id="a" * 32, message="", device=None)

    result = kokoro_adapter.synthesize(job, "This is a real Kokoro integration test.", voice.id)

    assert result["speech"].is_file()
    assert audio_io.verify_wav(result["speech"], require_signal=True) > 0


def test_real_seed_vc_conversion_when_installed(tmp_path, monkeypatch):
    available, detail = seed_vc_adapter.status()
    if not available:
        pytest.skip(detail)

    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    audio = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)[None, :].astype("float32") * 0.2
    audio_io.write_wav(source, audio, 16000)
    audio_io.write_wav(reference, audio[:, :8000], 16000)
    job = SimpleNamespace(job_id="b" * 32, message="", device=None)

    result = seed_vc_adapter.convert(job, source, reference)

    assert result["converted"].is_file()
    assert audio_io.verify_wav(result["converted"], require_signal=True) > 0
