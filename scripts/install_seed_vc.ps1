# Installs Seed-VC dependencies without replacing the backend CUDA-enabled PyTorch.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Vocal Studio - Seed-VC isolated inference environment"
Write-Host "This shares the existing .venv PyTorch installation and keeps Seed-VC's other packages isolated."
$confirm = Read-Host "Type YES to continue"
if ($confirm -ne "YES") { throw "Aborted." }

$mainPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $mainPython)) { throw "Create the backend .venv and install its CUDA PyTorch first." }
$cudaAvailable = & $mainPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cudaAvailable.Trim() -ne "True") {
  throw "The backend .venv cannot use CUDA. Fix its existing PyTorch installation first."
}

$project = Join-Path $Root "backend\seed-vc-project"
if (-not (Test-Path (Join-Path $project "inference.py"))) {
  Write-Host "Cloning official Seed-VC..."
  git clone https://github.com/Plachtaa/seed-vc.git $project
  if ($LASTEXITCODE -ne 0) { throw "Could not clone the official Seed-VC repository." }
} else {
  Write-Host "Official Seed-VC checkout already exists."
}

$venv = Join-Path $Root "backend\.venv-seed-vc"
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
  & py -3.12 -m venv --system-site-packages $venv
  if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.12 Seed-VC environment." }
}

$venvConfig = Join-Path $venv "pyvenv.cfg"
$configText = Get-Content $venvConfig -Raw
if ($configText -match "(?m)^include-system-site-packages\s*=") {
  $configText = [regex]::Replace(
    $configText,
    "(?m)^include-system-site-packages\s*=.*$",
    "include-system-site-packages = true"
  )
} else {
  $configText += "`ninclude-system-site-packages = true`n"
}
Set-Content -Path $venvConfig -Value $configText -Encoding ASCII

Write-Host "Installing Seed-VC inference requirements (PyTorch is not reinstalled)..."
& $python -m pip install -r (Join-Path $Root "backend\requirements-seed-vc.txt")
if ($LASTEXITCODE -ne 0) { throw "Seed-VC dependency installation failed." }

& $python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw "The Seed-VC environment cannot access the existing CUDA PyTorch." }

Write-Host ""
Write-Host "Explicitly install model assets (large download, confirmation required):"
Write-Host "  .venv\Scripts\python.exe scripts\download_models.py seed-vc"
Write-Host "Configure SEED_VC_PYTHON=backend/.venv-seed-vc/Scripts/python.exe in backend\.env"
Write-Host "Restart uvicorn after installing model assets. See docs\adapters\seed-vc.md"
