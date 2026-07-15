#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./install.sh --timesfm
.venv/bin/python - <<'PY'
from meteor_quant.timesfm.runtime import timesfm_capabilities
print(timesfm_capabilities())
PY
