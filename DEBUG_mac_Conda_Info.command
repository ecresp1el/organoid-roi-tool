#!/usr/bin/env bash
set -euo pipefail
echo "=== Debug: Conda & YAML ==="
command -v conda || { echo "[error] conda not on PATH"; exit 1; }
conda info
echo
echo "--- conda env list ---"
conda env list
echo
echo "--- environment.yml ---"
cat "$(dirname "$0")/environment.yml" || true
