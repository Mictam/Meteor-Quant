#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m meteor_quant.cli prepare-data --data-dir ./data "$@"
