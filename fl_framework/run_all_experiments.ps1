$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = 'python'
$moduleArgs = @('-m', 'fl_framework.run_paper_experiments')
$dataDir = Join-Path $scriptDir 'fl_framework\data'

Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Running federated learning experiments' -ForegroundColor Cyan
Write-Host 'Project root:' $scriptDir -ForegroundColor Cyan
Write-Host 'Data dir:' $dataDir -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan

& $python @moduleArgs --data-dir $dataDir
if ($LASTEXITCODE -ne 0) {
    throw "Experiment run failed with exit code $LASTEXITCODE"
}

Write-Host '========================================' -ForegroundColor Green
Write-Host 'All experiments finished successfully.' -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
