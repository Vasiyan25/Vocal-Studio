# RVC voice conversion

Vocal Studio launches the official RVC project's offline inference CLI in a dedicated Python process.
The RVC environment is isolated from the app's Python 3.12 / PyTorch 2.11 CUDA 12.8 environment: the
reviewed upstream RVC setup pins Python 3.12 and PyTorch 2.7.1, so installing its dependencies into the
app environment would risk breaking Demucs and Piper. No RVC runtime or model is bundled with this app.

## Reviewed runtime

- Upstream source: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Reviewed source revision: `81eed5e8f68b6bed1789f682fe78cdd324495afc`
- Official Windows/CUDA 12.8 dependency manifest: `requirments_cu128_py312.txt` in that revision.
- Official manifest targets Python 3.12 x64 and PyTorch below 2.8; its install recipe uses PyTorch and
  torchaudio `2.7.1+cu128`.
- The app's `requirements-rvc.txt` pins the smaller inference dependency set used by the official CLI.
  It excludes the training and Gradio UI dependencies.

The app runs the reviewed `infer/cli.py` through `rvc_runner.py`. It sets Hugging Face and Transformers
offline flags; the model checkpoint, optional index, HuBERT files, and RMVPE weights must already exist
locally. It does not use `rvc-python` (that package downloads auxiliary weights from its constructor).

## Install in an isolated environment (PowerShell)

Run from the project root. Python 3.12 is already required by the main app, but this creates a separate
environment and does not modify its installed packages:

```powershell
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git backend\rvc-project
git -C backend\rvc-project checkout 81eed5e8f68b6bed1789f682fe78cdd324495afc
py -3.12 -m venv backend\.venv-rvc
backend\.venv-rvc\Scripts\python.exe -m pip install --upgrade pip
backend\.venv-rvc\Scripts\python.exe -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
backend\.venv-rvc\Scripts\python.exe -m pip install -r backend\requirements-rvc.txt
```

These commands install packages and clone upstream code; review them before running. The PyTorch CUDA
wheels and dependency packages are a substantial download. They are deliberately not installed
automatically by Vocal Studio. CPU conversion can be selected with `RVC_DEVICE=cpu`; the same isolated
environment still needs the RVC CPU-capable dependencies.

The RVC inference CLI also requires FFmpeg and FFprobe on `PATH`. Install FFmpeg separately and restart
the backend after updating `PATH`.

## Windows native-extension / Application Control errors

The verified Python 3.12 environment may resolve Numba `0.68.0` with llvmlite `0.50.0` and NumPy
`1.26.4`. These versions satisfy Numba's declared Python, NumPy, and llvmlite constraints. A Windows
`DLL load failed` for `numba._devicearray` is not, by itself, evidence of a dependency conflict.

Check the actual Windows Code Integrity events and the extension signature without changing policy:

```powershell
Get-WinEvent -FilterHashtable @{
  LogName = "Microsoft-Windows-CodeIntegrity/Operational"
  Id = 3077,3033
  StartTime = (Get-Date).AddDays(-7)
} | Where-Object { $_.Message -match "numba|llvmlite|devicearray" } |
  Select-Object TimeCreated, Id, Message
Get-AuthenticodeSignature backend\.venv-rvc\Lib\site-packages\numba\_devicearray.cp312-win_amd64.pyd
```

If Code Integrity identifies an enforced application-control policy, changing Numba versions or renaming
the DLL is not an appropriate workaround. Ask the device administrator to verify the package provenance
and authorize the reviewed binary under the organization's approved signing/publisher policy. Do not
disable Smart App Control/WDAC or weaken the policy locally. The runtime readiness check reports captured
dependency versions and import diagnostics; RVC is not confirmed operational until a real conversion
completes and its WAV output validates.

## Explicitly acquire local inference assets

The upstream README documents HuBERT and RMVPE assets in the Hugging Face repository
[lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI). The following revision
was checked on 2026-10-21: `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`. Its repository card reports MIT;
the reviewed file sizes and SHA-256 values are provided for integrity checks. These are separate from
user-trained voice-model rights.

Use the `hf` command in the workspace's main `.venv`, or another trusted Hugging Face CLI installation.
This downloads only the listed RVC runtime assets; inspect the source and license first:

```powershell
\.venv\Scripts\hf.exe download lj1995/VoiceConversionWebUI hubert_base/config.json hubert_base/preprocessor_config.json hubert_base/pytorch_model.bin --revision e6d0c1a17da07c33557852f9dfa2bd44cc75737d --local-dir backend\rvc-project\assets
\.venv\Scripts\hf.exe download lj1995/VoiceConversionWebUI rmvpe.pt --revision e6d0c1a17da07c33557852f9dfa2bd44cc75737d --local-dir backend\rvc-project\assets\rmvpe
```

Checked artifact sizes and hashes:

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `hubert_base/pytorch_model.bin` | 189206711 | `cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5` |
| `rmvpe.pt` | 181184272 | `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` |

Verify after downloading:

```powershell
$hubertHash = (Get-FileHash backend\rvc-project\assets\hubert_base\pytorch_model.bin -Algorithm SHA256).Hash.ToLower()
$rmvpeHash = (Get-FileHash backend\rvc-project\assets\rmvpe\rmvpe.pt -Algorithm SHA256).Hash.ToLower()
if ($hubertHash -ne "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5") { throw "HuBERT SHA-256 mismatch" }
if ($rmvpeHash -ne "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193") { throw "RMVPE SHA-256 mismatch" }
```

HuBERT config files are small and are also required. The backend checks that both config files, HuBERT
weights, and `assets\rmvpe\rmvpe.pt` exist before reporting RVC ready. Normal inference is offline.

## Add an authorized trained voice

Put an inference-ready RVC v1 or v2 voice-conversion checkpoint in the app's model directory (not
the RVC source checkout):

```text
models\rvc\my-authorized-voice.pth
models\rvc\my-authorized-voice.index   # optional matching index
```

Use an ASCII filename stem containing only letters, numbers, dots, underscores, or hyphens. The `.pth`
must be the inference checkpoint format accepted by the pinned official CLI, not an arbitrary PyTorch
state dictionary: it must contain a `weight` mapping with `weight["emb_g.weight"]` and the RVC
`config` required to construct the synthesizer. RVC v1/v2 checkpoints may also specify `version` and
`f0`; the official loader uses legacy defaults if those keys are absent. Single- and multi-speaker
checkpoints are supported.

A matching FAISS added-index file named with the same stem and `.index` is optional. If there is no
usable index, conversion sets the index rate to zero and disables retrieval. Indexes must be generated
for the selected checkpoint/speaker; an unrelated or training index is not a valid substitute.

Voice discovery only exposes non-empty regular `.pth` files with safe ASCII IDs; it deliberately does
not load checkpoint contents while listing voices. The official CLI checks the checkpoint structure
when conversion is submitted. Checkpoints use PyTorch pickle serialization, which can execute code
when loaded: only use files from sources you trust and are authorized to use.

The software licence does not grant rights to a trained voice or the person whose voice it resembles.
Only use checkpoints and source recordings you are authorized to use, and follow their separate terms.

## Configuration, readiness, and conversion

Set these values in `backend\.env`:

```text
RVC_PYTHON=backend/.venv-rvc/Scripts/python.exe
RVC_PROJECT_DIR=backend/rvc-project
RVC_DEVICE=auto
```

`RVC_DEVICE` accepts `auto`, `cuda`, or `cpu`. `auto` uses CUDA if the isolated PyTorch runtime detects it,
otherwise the official inference code runs on CPU. The app verifies the Python version, PyTorch version,
runtime imports, FFmpeg/FFprobe, official checkout, local HuBERT/RMVPE assets, and at least one non-empty
checkpoint before enabling conversion. The `/api/voices` endpoint returns only safe IDs discovered from
those local files. A matching ID is revalidated when a conversion request is submitted, and the runner
accepts only that ID, not a client-supplied file path. The pinned CLI validates checkpoint compatibility
as it loads the selected file. Conversion results are exposed as job status plus WAV preview and download
endpoints.

Restart the backend and open Voice Conversion. Readiness is also visible from:

```powershell
curl.exe http://127.0.0.1:8000/api/models
curl.exe http://127.0.0.1:8000/api/voices
```

RVC inference can use substantial GPU memory. A 6 GB laptop GPU may require short inputs; on GPU memory
errors, try a shorter vocal stem or explicitly set `RVC_DEVICE=cpu`. Conversion runs through the existing
single-worker job queue, with a timeout, bounded logs, output validation, and playback/download endpoints.

## Tests and limitations

From the project root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -q
Set-Location ..\frontend
npm run build
```

The adapter tests mock the subprocess and output; they validate API/job flow, model selection, failure
handling, timeout, and malformed output. They do not demonstrate real RVC inference. A real conversion
requires the isolated runtime, local upstream assets, FFmpeg/FFprobe, and an authorized trained voice.
The installed app's main PyTorch/CUDA environment remains unchanged by the RVC setup.
