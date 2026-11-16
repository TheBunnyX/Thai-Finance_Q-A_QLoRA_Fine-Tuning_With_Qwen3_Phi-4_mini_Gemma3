# Speed-only benchmarks per model. Training: 20 optimizer steps at several micro batch sizes (effective batch stays 8).
# Evaluation: same 48-row subset at several batch sizes with the scale-v1 adapter (also records accuracy to catch drift).
param([string[]]$Models = @('qwen','phi','gemma'), [string]$Data = 'data/processed/finance-scale-steps-v1')
$ErrorActionPreference = 'Continue'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$py = Join-Path $rootPath '.venv\Scripts\python.exe'
foreach ($m in $Models) {
    foreach ($b in @(1,2,4)) {
        $dir = "outputs/bench/train-$m-b$b"
        if (Test-Path -LiteralPath $dir) { continue }
        Write-Host "TRAIN bench $m micro-batch $b"
        & $py -X utf8 -m finance_qlora train --model $m --data $Data --run-dir $dir --batch-size $b --max-steps 20 --no-progress 2>&1 | Select-Object -Last 3
    }
    foreach ($b in @(4,8,16)) {
        $dir = "outputs/bench/eval-$m-b$b"
        if (Test-Path -LiteralPath $dir) { continue }
        Write-Host "EVAL bench $m batch $b"
        & $py -X utf8 -m finance_qlora evaluate --model $m --data data/processed/finance-scale-2500-v1 --split validation --limit 48 `
            --max-new-tokens 256 --warmup-batches 1 --batch-size $b --adapter "outputs/$m-scale-v1/adapter" --output $dir --no-progress 2>&1 | Select-Object -Last 3
    }
}
Write-Host 'Done: bench'
