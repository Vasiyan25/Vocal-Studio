"""Runs Demucs in its own process (so GPU memory is freed afterwards and a crash cannot take down the API).

Usage (done by demucs_adapter.py, from the backend folder):
    python -m app.services.adapters.demucs_runner INPUT OUTDIR MODEL DEVICE

Prints one JSON object per line on stdout. The last one has stage "done".
It only reads weights from the local cache; it never downloads anything.
"""
import json
import sys
from pathlib import Path


def emit(**kw) -> None:
    print(json.dumps(kw), flush=True)


def main() -> int:
    src, out_dir, model_name, device_pref = sys.argv[1:5]
    out = Path(out_dir)

    import torch
    from demucs.apply import apply_model
    from demucs.hf import get_hf_model  # loads from the local Hugging Face cache (offline), no AWS fallback

    from app.services import audio_io

    def pick_device() -> str:
        if device_pref == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if device_pref == "cuda":
            raise RuntimeError("DEMUCS_DEVICE=cuda was requested but PyTorch cannot see a CUDA GPU.")
        return "cpu"

    def label(dev: str) -> str:
        return f"cuda ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "cpu"

    device = pick_device()
    emit(message="Loading the separation model", device=label(device))
    model = get_hf_model(model_name)
    model.eval()

    emit(message="Decoding the audio")
    audio = audio_io.decode(Path(src), model.samplerate, model.audio_channels)
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    def run(dev: str):
        with torch.no_grad():
            return apply_model(model, wav[None], device=dev, split=True, overlap=0.25,
                               shifts=1, progress=False)[0].cpu()

    emit(message=f"Separating vocals on {label(device)}", device=label(device))
    try:
        sources = run(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        device = "cpu"
        emit(message="GPU ran out of memory, retrying on the CPU (slower)", device=label(device))
        sources = run(device)

    sources = sources * std + mean
    names = list(model.sources)
    vocals = sources[names.index("vocals")].numpy()
    instrumental = sum(sources[i] for i, n in enumerate(names) if n != "vocals").numpy()

    emit(message="Writing the output files")
    out.mkdir(parents=True, exist_ok=True)
    audio_io.write_wav(out / "vocals.wav", vocals, model.samplerate)
    audio_io.write_wav(out / "instrumental.wav", instrumental, model.samplerate)
    emit(stage="done", device=label(device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
