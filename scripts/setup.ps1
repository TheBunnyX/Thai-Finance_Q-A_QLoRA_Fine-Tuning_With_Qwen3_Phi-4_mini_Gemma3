param([switch]$InstallTraining)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -c "import sys; assert sys.version_info[:3] == (3,12,10), 'Python 3.12.10 required'"
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12.10 first' }
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv' }
}
& $pythonPath -c "import sys; assert sys.version_info[:3] == (3,12,10), 'Wrong .venv Python version'"
if ($LASTEXITCODE -ne 0) { throw 'Existing .venv must use Python 3.12.10' }
if ($InstallTraining) {
    # Preserve the tested CUDA torch build; fail resolution rather than silently replacing it.
    $torchVersion = & $pythonPath -c "import importlib.metadata; print(importlib.metadata.version('torch'))"
    if ($LASTEXITCODE -ne 0) { throw 'Install CUDA-enabled PyTorch first' }
    New-Item -ItemType Directory -Force -Path 'reports' | Out-Null
    $constraintPath = Join-Path $projectRoot 'reports\torch-constraint.txt'
    Set-Content -LiteralPath $constraintPath -Value "torch==$torchVersion" -Encoding ascii
    & $pythonPath -m pip install -r requirements-training-windows.txt -c $constraintPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Training dependency installation failed. Review compatibility; do not replace CUDA torch blindly. See README for Windows/WSL options.'
    }
    & $pythonPath -m finance_qlora doctor --training --output reports/environment-training.json
} else {
    & $pythonPath -m finance_qlora doctor
}
if ($LASTEXITCODE -ne 0) { throw 'Environment checks failed; inspect reports/environment*.json' }
