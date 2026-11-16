param([string]$Data = 'data/processed/finance-dev-1250-v3')
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$pythonPath = Join-Path $rootPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath "$Data/manifest.json")) { throw 'Missing development dataset manifest.' }
foreach ($modelKey in @('qwen','phi','gemma')) {
    $tokensFile = "reports/tokens-$modelKey-dev-v3.json"
    if (-not (Test-Path -LiteralPath $tokensFile)) { throw "Tokenizer check missing: $tokensFile" }
    $tokens = Get-Content -LiteralPath $tokensFile -Raw | ConvertFrom-Json
    foreach ($splitName in @('train','validation','test')) {
        if ($tokens.splits.$splitName.failures.Count -ne 0) { throw "Tokenizer errors: $modelKey/$splitName" }
    }
    $smokeFile = "outputs/$modelKey-1250-smoke/run.json"
    if (-not (Test-Path -LiteralPath $smokeFile)) { throw "Smoke run missing: $smokeFile" }
    $smoke = Get-Content -LiteralPath $smokeFile -Raw | ConvertFrom-Json
    if ($smoke.status -ne 'complete') { throw "Smoke run incomplete: $modelKey" }
    if (Test-Path -LiteralPath "outputs/$modelKey-dev-v3") { throw "Output exists: outputs/$modelKey-dev-v3" }
}
foreach ($modelKey in @('phi','gemma','qwen')) {
    $batch = if ($modelKey -in @('phi','gemma')) { 1 } else { 2 }
    Write-Host "Training $modelKey on development v3; micro batch $batch, effective batch 8"
    & $pythonPath -m finance_qlora train --model $modelKey --data $Data --run-dir "outputs/$modelKey-dev-v3" --batch-size $batch --progress-log "reports/$modelKey-dev-v3-progress.jsonl"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $modelKey with exit code $LASTEXITCODE" }
}
Write-Host 'All four development training runs completed.'
