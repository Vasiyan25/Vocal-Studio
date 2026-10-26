"""Subprocess entrypoint that runs official Seed-VC inference.py and relocates the WAV.

This keeps Seed-VC imports inside the isolated environment and outside the FastAPI process.
"""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Seed-VC inference for Vocal Studio")
    parser.add_argument("--seed-vc-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, help="Final converted.wav path")
    parser.add_argument("--work-dir", required=True, help="Temporary directory for Seed-VC output")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--fp16", default="true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.seed_vc_root).resolve()
    source = Path(args.source).resolve()
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()

    if not (root / "inference.py").is_file():
        raise SystemExit(f"Seed-VC inference.py not found under {root}")
    if not source.is_file():
        raise SystemExit("Source audio file is missing.")
    if not reference.is_file():
        raise SystemExit("Reference audio file is missing.")
    if not checkpoint.is_file():
        raise SystemExit("Seed-VC checkpoint file is missing.")
    if not config.is_file():
        raise SystemExit("Seed-VC config file is missing.")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the adapter's shared asset cache, with a per-job fallback for direct runner use.
    os.environ.setdefault("HF_HUB_CACHE", str((work_dir / "hf_cache").resolve()))

    sys.path.insert(0, str(root))
    os.chdir(root)

    sys.argv = [
        str(root / "inference.py"),
        "--source", str(source),
        "--target", str(reference),
        "--output", str(work_dir),
        "--diffusion-steps", str(args.diffusion_steps),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7",
        "--f0-condition", "False",
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", "0",
        "--checkpoint", str(checkpoint),
        "--config", str(config),
        "--fp16", str(args.fp16),
    ]
    try:
        runpy.run_path(str(root / "inference.py"), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        if code != 0:
            raise SystemExit(code) from None

    produced = sorted(work_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not produced:
        # Some versions may write a single wav without the vc_ prefix.
        produced = sorted(work_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not produced:
        raise RuntimeError("Seed-VC finished without generating a WAV file.")
    shutil.copyfile(produced[0], output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Seed-VC finished without writing a usable output file.")
    print(f"SEED_VC_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
