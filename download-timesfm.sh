#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("google/timesfm-2.5-200m-pytorch")
print("TimesFM model cached at", path)
PY
