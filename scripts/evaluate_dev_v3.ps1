param([string]$Data = 'data/processed/finance-dev-1250-v3')
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$pythonPath = Join-Path $rootPath '.venv\Scripts\python.exe'
foreach ($modelKey in @('qwen','phi','gemma')) {
    $runFile = "outputs/$modelKey-dev-v3/run.json"
    if (-not (Test-Path -LiteralPath $runFile)) { throw "Run missing: $runFile" }
    $run = Get-Content -LiteralPath $runFile -Raw | ConvertFrom-Json
    if ($run.status -ne 'complete') { throw "Run incomplete: $modelKey" }
    foreach ($variant in @('base','sft')) {
        if (Test-Path -LiteralPath "outputs/eval-$modelKey-dev-v3-validation-$variant") { throw "Output already exists for $modelKey/$variant" }
    }
}
foreach ($modelKey in @('phi','gemma','qwen')) {
    $batch = if ($modelKey -in @('phi','gemma')) { 2 } else { 4 }
    foreach ($variant in @('base','sft')) {
        $destination = "outputs/eval-$modelKey-dev-v3-validation-$variant"
        $argsList = @('evaluate','--model',$modelKey,'--data',$Data,'--split','validation','--max-new-tokens','256',
                      '--warmup-batches','1','--batch-size',"$batch",'--output',$destination,
                      '--progress-log',"reports/eval-$modelKey-dev-v3-validation-$variant-progress.jsonl")
        if ($variant -eq 'sft') { $argsList += @('--adapter',"outputs/$modelKey-dev-v3/adapter") }
        Write-Host "Evaluating $modelKey / $variant on development validation; batch $batch"
        & $pythonPath -m finance_qlora @argsList
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $modelKey/$variant with exit code $LASTEXITCODE" }
    }
    & $pythonPath -m finance_qlora compare "outputs/eval-$modelKey-dev-v3-validation-base/metrics.json" "outputs/eval-$modelKey-dev-v3-validation-sft/metrics.json" --output "reports/$modelKey-dev-v3-validation-comparison.json"
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed for $modelKey" }
}
Write-Host 'Validation comparisons complete. Test split remains unused.'
