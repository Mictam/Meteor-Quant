#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run-id>" >&2
  exit 2
fi
cd "$(dirname "$0")"
RUN_DIR="data/markethybrid/runs/$1"
if [[ ! -x .venv/bin/python ]]; then
  echo "Project virtual environment was not found. Run ./install.sh first." >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/spec.json" ]]; then
  echo "MarketHybrid run was not found: $RUN_DIR" >&2
  exit 1
fi
rm -f "$RUN_DIR/result.json"
exec .venv/bin/python -m meteor_quant.markethybrid.worker "$RUN_DIR"
