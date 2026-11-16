# Qwen-only grid, selected on the validation split only (test stays unused).
# Each config: train -> evaluate validation with a 512-token budget (step-by-step answers are longer).
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$py = Join-Path $rootPath '.venv\Scripts\python.exe'
$steps = 'data/processed/finance-scale-steps-v1'
$configs = @(
    @{ Run = 'qwen-steps-1ep';     Extra = @() },
    @{ Run = 'qwen-steps-2ep';     Extra = @('--epochs','2') },
    @{ Run = 'qwen-steps-2ep-r32'; Extra = @('--epochs','2','--lora-rank','32') }
)
foreach ($c in $configs) {
    if (Test-Path -LiteralPath "outputs/$($c.Run)") { Write-Host "Skip training, exists: $($c.Run)" }
    else {
        Write-Host "Training $($c.Run)"
        & $py -X utf8 -m finance_qlora train --model qwen --data $steps --run-dir "outputs/$($c.Run)" --batch-size 2 `
            --progress-log "reports/$($c.Run)-progress.jsonl" --no-progress @($c.Extra)
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $($c.Run)" }
    }
    $tag = $c.Run -replace '^qwen-', ''
    powershell -ExecutionPolicy Bypass -File scripts/evaluate_scale_v1.ps1 -Split validation -Models qwen -Variants scale `
        -Data $steps -Tag $tag -AdapterRun $c.Run.Substring(5) -MaxNewTokens 512
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($c.Run)" }
}
Write-Host 'Done: qwen experiments'
