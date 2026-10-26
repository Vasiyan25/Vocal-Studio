"""Run the official RVC inference CLI in offline mode from its separate environment."""
import argparse
import importlib
import os
import re
import sys
from pathlib import Path


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-root", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--speaker-id", type=int)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    if not VOICE_ID_PATTERN.fullmatch(args.voice_id):
        parser.error("Invalid voice model identifier.")
    project = args.rvc_root.resolve()
    models = args.models_dir.resolve()
    source = args.input.resolve()
    output = args.output.resolve()
    model = models / f"{args.voice_id}.pth"
    index = models / f"{args.voice_id}.index"
    if not (project / "infer" / "cli.py").is_file():
        parser.error("The official RVC inference CLI is missing.")
    if not model.is_file() or model.is_symlink() or model.stat().st_size == 0:
        parser.error("The selected voice checkpoint is missing or empty.")
    if not source.is_file():
        parser.error("Input audio is missing.")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    sys.path.insert(0, str(project))
    os.chdir(project)

    rvc_cli = importlib.import_module("infer.cli")
    speaker_count, speakers = rvc_cli.load_model_metadata(str(model.resolve()))
    speaker_id = rvc_cli.select_speaker(speaker_count, speakers, args.speaker_id)

    command = [
        "--model", str(model.resolve()),
        "--input", str(source),
        "--output", str(output),
        "--speaker-id", str(speaker_id),
        "--f0-method", "rmvpe",
        "--index-rate", "0.75" if index.is_file() and not index.is_symlink() and index.stat().st_size else "0",
        "--overwrite",
    ]
    if index.is_file() and not index.is_symlink() and index.stat().st_size:
        command.extend(["--index", str(index.resolve())])
    cli_status = rvc_cli.main(command)
    if cli_status not in (None, 0):
        raise RuntimeError(f"RVC CLI failed with exit status {cli_status}.")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("RVC CLI returned success without generating a non-empty output audio file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
