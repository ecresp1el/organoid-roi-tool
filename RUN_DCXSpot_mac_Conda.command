#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="organoid_roi_incucyte_imaging"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] Conda not found on PATH. Run 'conda init' and restart Terminal."
  exit 1
fi

BASE_DIR="$(conda info --base 2>/dev/null || echo "")"
ENV_DIR="${BASE_DIR}/envs/${ENV_NAME}"
if [ ! -d "$ENV_DIR" ]; then
  echo "[error] Conda env not found: $ENV_DIR"
  echo "[hint] Run ./RUN_FIRST_mac_Conda_Create_And_Launch.command once to create it."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/dcxspot_config.json"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[error] Config not found: $CONFIG_PATH"
  exit 1
fi

echo "[run] Using env: $ENV_DIR"
echo "[run] Config : $CONFIG_PATH"
conda run -n "$ENV_NAME" python -m dcxspot_play.run_configured
