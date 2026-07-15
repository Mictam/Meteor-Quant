$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m meteor_quant.cli prepare-data --data-dir .\data @args
