# Evaluate variants on finance-scale-2500-v1. Variants: base | dev-v3 | scale (adapter dirs outputs/<model>-<run>/adapter).
# validation is for choosing settings; test (independent of train) should be scored once per final variant.
param([ValidateSet('validation','test')][string]$Split = 'validation',
      [string[]]$Models = @('qwen','phi','gemma'),
      [string[]]$Variants = @('scale'),
      [string]$Data = 'data/processed/finance-scale-2500-v1',
      [string]$Tag = 'scale-v1', [string]$AdapterRun = 'scale-v1', [string]$MaxNewTokens = '256')
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$pythonPath = Join-Path $rootPath '.venv\Scripts\python.exe'
foreach ($modelKey in $Models) {
    $batch = if ($modelKey -in @('phi','gemma')) { 2 } else { 4 }
    foreach ($variant in $Variants) {
        $destination = "outputs/eval-$modelKey-$Tag-$Split-$variant"
        if (Test-Path -LiteralPath $destination) { Write-Host "Skip existing $destination"; continue }
        $argsList = @('evaluate','--model',$modelKey,'--data',$Data,'--split',$Split,'--max-new-tokens',$MaxNewTokens,
                      '--warmup-batches','1','--batch-size',"$batch",'--output',$destination,
                      '--progress-log',"reports/eval-$modelKey-$Tag-$Split-$variant-progress.jsonl",'--no-progress')
        if ($variant -eq 'dev-v3') { $argsList += @('--adapter',"outputs/$modelKey-dev-v3/adapter") }
        if ($variant -eq 'scale') { $argsList += @('--adapter',"outputs/$modelKey-$AdapterRun/adapter") }
        Write-Host "Evaluating $modelKey / $variant on $Split; batch $batch"
        & $pythonPath -X utf8 -m finance_qlora @argsList
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $modelKey/$variant with exit code $LASTEXITCODE" }
    }
}
Write-Host "Done: $Split"
