param([switch]$Install)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$webPath = Join-Path $projectRoot 'web'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the project .venv first.' }
if ($Install) {
    Push-Location -LiteralPath $webPath
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $webPath 'node_modules'))) { throw 'Run with -Install for the first launch.' }
$existingPorts = Get-NetTCPConnection -LocalPort 8000,3000 -State Listen -ErrorAction SilentlyContinue
if ($existingPorts) { throw 'Port 8000 or 3000 is in use. Stop the existing playground first.' }
# Both children are hidden and belong to this launcher; Ctrl+C cleans them up.
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','finance_qlora.playground') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
try {
    $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
    $nextPath = Join-Path $webPath 'node_modules\next\dist\bin\next'
    $webProcess = Start-Process -FilePath $nodePath -ArgumentList @("`"$nextPath`"",'dev','--hostname','127.0.0.1','--port','3000') -WorkingDirectory $webPath -WindowStyle Hidden -PassThru
    Write-Host 'Finance Lab: http://127.0.0.1:3000'
    Write-Host 'API: http://127.0.0.1:8000/status | Ctrl+C stops both services.'
    while (-not $apiProcess.HasExited -and -not $webProcess.HasExited) { Start-Sleep -Seconds 1 }
    throw 'A playground service exited. Run the two commands in docs/playground.md to see its error.'
} finally {
    if ($webProcess -and -not $webProcess.HasExited) { Stop-Process -Id $webProcess.Id -ErrorAction SilentlyContinue }
    if (-not $apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id -ErrorAction SilentlyContinue }
}
