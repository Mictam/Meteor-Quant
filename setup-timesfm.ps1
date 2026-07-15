param([switch]$DownloadModel)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\install.ps1 --timesfm
if ($LASTEXITCODE -ne 0) { throw "TimesFM installation failed" }
& .\.venv\Scripts\python.exe -c "from meteor_quant.timesfm.runtime import timesfm_capabilities; print(timesfm_capabilities())"
if ($DownloadModel) { & .\download-timesfm.ps1 }
