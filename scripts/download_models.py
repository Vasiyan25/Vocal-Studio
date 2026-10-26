"""Download model files ONLY after you confirm. Nothing is ever downloaded automatically.

Run from the project folder with the backend venv active:
    python scripts/download_models.py demucs
    python scripts/download_models.py piper
    python scripts/download_models.py kokoro --voice af_heart
    python scripts/download_models.py seed-vc
Add --yes to skip the question (for scripts).
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

INFO = {
    "demucs": (
        "Demucs htdemucs weights (safetensors) from Hugging Face repo adefossez/HTDemucs.\n"
        "  Code licence: MIT. WEIGHTS licence: unresolved; the author has said the weights are for scientific use.\n"
        "  Use for personal/portfolio work only until you have written clarification. See docs/adapters/demucs.md."
    ),
    "piper": (
        "Piper voice en_US-lessac-medium (.onnx + .json) from Hugging Face repo rhasspy/piper-voices.\n"
        "  The voice has its own licence (Blizzard 2013 Lessac dataset):\n"
        "  https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html\n"
        "  Read it before commercial use. See docs/adapters/piper.md."
    ),
    "kokoro": (
        "Kokoro-82M model and English voice packs from Hugging Face repo hexgrad/Kokoro-82M.\n"
        "  Model weights: Apache-2.0. Voice packs are model assets; review model-card and training-data attributions."
    ),
    "seed-vc": (
        "Seed-VC offline speech-conversion checkpoint/config and Whisper-small, BigVGAN, and CAMPPlus assets.\n"
        "  Seed-VC code and checkpoint: GPL-3.0. Whisper-small: Apache-2.0. Review the BigVGAN and CAMPPlus model cards."
    ),
}


def confirm(name: str, size_note: str, assume_yes: bool) -> None:
    print(f"\nAbout to download into {MODELS}:\n  {INFO[name]}\n  Size: {size_note}")
    if assume_yes:
        return
    if input("\nType 'yes' to download, anything else to cancel: ").strip().lower() != "yes":
        print("Cancelled. Nothing was downloaded.")
        sys.exit(0)


def demucs(assume_yes: bool) -> None:
    os.environ["HF_HOME"] = str(MODELS / "hf")  # must be set before huggingface_hub is imported
    from huggingface_hub import HfApi, hf_hub_download
    import yaml

    repo = "adefossez/HTDemucs"
    try:
        files = HfApi().model_info(repo, files_metadata=True).siblings
        total = sum((f.size or 0) for f in files if f.rfilename.endswith((".safetensors", ".yaml")))
        size_note = f"{total / 1048576:.0f} MB (reported by Hugging Face)"
    except Exception:
        size_note = "unknown (could not ask Hugging Face)"
    confirm("demucs", size_note, assume_yes)
    cfg = hf_hub_download(repo, "htdemucs.yaml")
    for sig in yaml.safe_load(open(cfg, encoding="utf-8"))["models"]:
        print("Downloading", f"{sig}.safetensors", "...")
        hf_hub_download(repo, f"{sig}.safetensors")
    print("Done. Restart the backend; Settings should now show Demucs as ready.")


def piper(assume_yes: bool) -> None:
    voice = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
    confirm("piper", "shown by the downloader", assume_yes)
    target = MODELS / "piper"
    subprocess.run([sys.executable, "-m", "piper.download_voices", "--download-dir", str(target), voice], check=True)
    print("Done. Restart the backend; Settings should now show Piper as ready.")


def kokoro(assume_yes: bool, voices: list[str]) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo = "hexgrad/Kokoro-82M"
    filenames = ["config.json", "kokoro-v1_0.pth", *(f"voices/{voice}.pt" for voice in voices)]
    metadata = HfApi().model_info(repo, files_metadata=True).siblings
    sizes = {item.rfilename: item.size or 0 for item in metadata}
    missing = [name for name in filenames if name not in sizes]
    if missing:
        raise RuntimeError(f"Files not found in the official Kokoro repository: {', '.join(missing)}")
    total = sum(sizes[name] for name in filenames)
    confirm("kokoro", f"{total / 1048576:.0f} MB (reported by Hugging Face)", assume_yes)
    target = MODELS / "kokoro"
    target.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print("Downloading", filename, "...")
        hf_hub_download(repo, filename, local_dir=str(target))
    print("Done. Restart the backend; Kokoro and the selected voice(s) should now be available.")


def seed_vc(assume_yes: bool) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    files = {
        "Plachta/Seed-VC": (
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        ),
        "funasr/campplus": ("campplus_cn_common.bin",),
    }
    repos = ("openai/whisper-small", "nvidia/bigvgan_v2_22khz_80band_256x")
    api = HfApi()
    total = 0
    for repo, filenames in files.items():
        metadata = api.model_info(repo, files_metadata=True).siblings
        sizes = {item.rfilename: item.size or 0 for item in metadata}
        missing = [name for name in filenames if name not in sizes]
        if missing:
            raise RuntimeError(f"Files not found in {repo}: {', '.join(missing)}")
        total += sum(sizes[name] for name in filenames)
    for repo in repos:
        total += sum(item.size or 0 for item in api.model_info(repo, files_metadata=True).siblings)
    confirm("seed-vc", f"approximately {total / 1073741824:.2f} GiB (reported by Hugging Face)", assume_yes)

    target = MODELS / "seed-vc"
    cache = target / "hf_cache"
    target.mkdir(parents=True, exist_ok=True)
    for repo, filenames in files.items():
        for filename in filenames:
            print("Downloading", f"{repo}/{filename}", "...")
            downloaded = Path(hf_hub_download(
                repo,
                filename,
                local_dir=str(target) if repo == "Plachta/Seed-VC" else None,
                cache_dir=None if repo == "Plachta/Seed-VC" else str(cache),
            ))
            if repo == "Plachta/Seed-VC" and filename.endswith(".pth"):
                if downloaded.stat().st_size != 440_312_082:
                    raise RuntimeError("Seed-VC checkpoint size does not match the official repository metadata.")
                digest = hashlib.sha256()
                with downloaded.open("rb") as checkpoint:
                    for block in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != "8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720":
                    raise RuntimeError("Seed-VC checkpoint SHA-256 does not match the official model.")
    for repo in repos:
        print("Downloading auxiliary model", repo, "...")
        snapshot_download(repo, cache_dir=str(cache))
    print("Done. Restart the backend; Seed-VC will run with network downloads disabled.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["demucs", "piper", "kokoro", "seed-vc"])
    ap.add_argument("--yes", action="store_true", help="skip the confirmation question")
    ap.add_argument(
        "--voice",
        action="append",
        help="Kokoro English voice id to install; repeat to add voices (default: af_heart).",
    )
    a = ap.parse_args()
    if a.model == "kokoro":
        selected = a.voice or ["af_heart"]
        if any(not re.fullmatch(r"(?:af|am|bf|bm)_[a-z]+", voice) for voice in selected):
            ap.error("Kokoro voice IDs must be English IDs such as af_heart or bm_fable.")
        kokoro(a.yes, selected)
    elif a.model == "seed-vc":
        seed_vc(a.yes)
    else:
        {"demucs": demucs, "piper": piper}[a.model](a.yes)
