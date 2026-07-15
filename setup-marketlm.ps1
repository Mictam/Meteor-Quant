$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\install.ps1 --marketlm
if ($LASTEXITCODE -ne 0) { throw "MarketLM installation failed" }
& .\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
