"""Stand-in for demucs_runner used ONLY to test the adapter's process handling (parsing, errors, timeout).
It does no AI. Mode comes from the FAKE_MODE environment variable."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from app.services import audio_io

mode = os.environ.get("FAKE_MODE", "ok")
out = Path(sys.argv[2])
print(json.dumps({"message": "Fake separating", "device": "cpu"}), flush=True)
if mode == "hang":
    time.sleep(30)
if mode == "crash":
    print("Traceback: boom at C:\\secret\\place", file=sys.stderr)
    sys.exit(1)
t = np.linspace(0, 1, 8000, endpoint=False)
audio_io.write_wav(out / "vocals.wav", np.sin(2 * np.pi * 220 * t)[None, :].astype("float32") * 0.5, 8000)
if mode != "missing_instrumental":
    audio_io.write_wav(out / "instrumental.wav", np.sin(2 * np.pi * 110 * t)[None, :].astype("float32") * 0.5, 8000)
print(json.dumps({"stage": "done", "device": "cpu"}), flush=True)
