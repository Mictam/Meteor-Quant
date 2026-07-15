param(
    [Parameter(Mandatory = $true)]
    [string]$RunId
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunDir = Join-Path $ProjectRoot "data\markethybrid\runs\$RunId"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}
if (-not (Test-Path (Join-Path $RunDir "spec.json"))) {
    throw "MarketHybrid run was not found: $RunDir"
}

$Checkpoints = Get-ChildItem -Path $RunDir -Filter "checkpoint_step_*.pt" -ErrorAction SilentlyContinue |
    Sort-Object Name
if ($Checkpoints.Count -gt 0) {
    Write-Host "Resuming from $($Checkpoints[-1].Name)"
} else {
    Write-Host "No periodic checkpoint exists; the run will restart training from step 0 and reuse prepared tensors."
}

Remove-Item (Join-Path $RunDir "result.json") -Force -ErrorAction SilentlyContinue
& $Python -m meteor_quant.markethybrid.worker $RunDir
exit $LASTEXITCODE
