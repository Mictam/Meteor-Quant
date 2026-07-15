#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cd frontend
npm ci --no-audit --no-fund
npm run build
cd ..
PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="${PYTHON:-python3}"; fi
"$PYTHON_BIN" scripts/sync_frontend.py
