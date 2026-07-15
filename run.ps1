param(
  [int]$Port = 8000,
  [string]$HostAddress = "127.0.0.1"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv\Scripts\python.exe)) {
  throw "Virtual environment is missing. Run .\install.ps1 first."
}
& .\.venv\Scripts\python.exe -m meteor_quant.cli serve --host $HostAddress --port $Port
