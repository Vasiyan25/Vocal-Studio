"""Prints what this Python environment can really do. Run with the backend venv active:
    python scripts/check_env.py
"""
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

print("Python  :", platform.python_version(), sys.executable)
try:
    import torch
    print("PyTorch :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU     :", p.name, f"({p.total_memory / 1073741824:.1f} GB)")
except ImportError:
    print("PyTorch : not installed")
for exe in ("ffmpeg", "ffprobe"):
    print(f"{exe:8}:", shutil.which(exe) or "NOT FOUND on PATH")
from app.services.model_registry import list_models  # noqa: E402

print()
for m in list_models():
    print(f"[{'ready' if m.available else 'not ready'}] {m.name} ({m.feature}): {m.detail}")
