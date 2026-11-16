# Same recipe as dev-v3 (1 epoch, r16, LR 1e-4); only the training data changes (2500 rows).
param([string]$Data = 'data/processed/finance-scale-2500-v1', [string[]]$Models = @('qwen','phi','gemma'))
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$pythonPath = Join-Path $rootPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath "$Data/manifest.json")) { throw 'Missing dataset manifest.' }
foreach ($modelKey in $Models) {
    if (Test-Path -LiteralPath "outputs/$modelKey-scale-v1") { throw "Output exists: outputs/$modelKey-scale-v1" }
}
foreach ($modelKey in $Models) {
    $batch = if ($modelKey -in @('phi','gemma')) { 1 } else { 2 }
    Write-Host "Training $modelKey on scale-2500-v1; micro batch $batch, effective batch 8"
    & $pythonPath -X utf8 -m finance_qlora train --model $modelKey --data $Data --run-dir "outputs/$modelKey-scale-v1" --batch-size $batch --progress-log "reports/$modelKey-scale-v1-progress.jsonl" --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $modelKey with exit code $LASTEXITCODE" }
}
Write-Host 'Scale training runs completed.'
