#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./install.sh --markethybrid
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if not torch.cuda.is_available():
    print("warning: practical MarketHybrid training normally requires a CUDA GPU")
PY
