param(
    [string]$Data = 'data/processed/finance-best-1250-v1',
    [string]$Round = '1250-v1',
    [int]$BatchSize = 4,
    [int]$ProfileSamples = 10
)
$ErrorActionPreference = 'Stop'
if ($Round -notmatch '^[a-zA-Z0-9-]+$' -or $BatchSize -lt 1 -or $ProfileSamples -lt 1) {
    throw 'Invalid round name, batch size or profile size'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
function Invoke-FinanceCommand {
    param([string[]]$JobArguments)
    & $pythonPath -m finance_qlora @JobArguments
    if ($LASTEXITCODE -ne 0) { throw "Evaluation command failed: $($JobArguments -join ' ')" }
}
foreach ($modelKey in @('qwen')) {
    $runPath = "outputs/$modelKey-$Round"
    $run = Get-Content -LiteralPath "$runPath/run.json" -Raw | ConvertFrom-Json
    if ($run.status -ne 'complete') { throw "Training is incomplete: $runPath" }
}
$predictionPaths = @()
foreach ($modelKey in @('qwen')) {
    foreach ($variant in @('base','sft')) {
        $destination = "outputs/eval-$modelKey-$Round-$variant"
        $common = @('evaluate','--model',$modelKey,'--data',$Data,'--split','test','--max-new-tokens','256','--warmup-batches','1')
        if ($variant -eq 'sft') { $common += @('--adapter',"outputs/$modelKey-$Round/adapter") }
        Invoke-FinanceCommand -JobArguments ($common + @('--batch-size',"$BatchSize",'--output',$destination,'--progress-log',"reports/eval-$modelKey-$Round-$variant-progress.jsonl"))
        $predictionPaths += "$destination/predictions.jsonl"
        Invoke-FinanceCommand -JobArguments ($common + @('--batch-size','1','--measure-ttft','--limit',"$ProfileSamples",'--selection-seed','42','--output',"$destination-profile",'--progress-log',"reports/eval-$modelKey-$Round-$variant-profile-progress.jsonl"))
    }
    foreach ($suffix in @('','-profile')) {
        Invoke-FinanceCommand -JobArguments @('compare',"outputs/eval-$modelKey-$Round-base$suffix/metrics.json", "outputs/eval-$modelKey-$Round-sft$suffix/metrics.json",'--output',"reports/$modelKey-$Round-comparison$suffix.json")
    }
}
Invoke-FinanceCommand -JobArguments (@('review-export') + $predictionPaths + @('--output',"outputs/review-$Round"))
