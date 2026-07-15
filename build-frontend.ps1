$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Push-Location frontend
try {
  npm ci --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
  npm run build
  if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
} finally {
  Pop-Location
}
$Python = if (Test-Path .venv\Scripts\python.exe) { ".venv\Scripts\python.exe" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if ($Python -eq "py") { & py -3.11 scripts\sync_frontend.py } else { & $Python scripts\sync_frontend.py }
if ($LASTEXITCODE -ne 0) { throw "frontend package sync failed" }
