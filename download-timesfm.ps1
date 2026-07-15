$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run .\install.ps1 and .\setup-timesfm.ps1 first." }
& $Python -c "from huggingface_hub import snapshot_download; path=snapshot_download('google/timesfm-2.5-200m-pytorch'); print('TimesFM model cached at', path)"
if ($LASTEXITCODE -ne 0) { throw "TimesFM model download failed" }
