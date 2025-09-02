#!/usr/bin/env bash
set -euo pipefail

# Generate synthetic TIFFs and reorganize into a sample project.
# Usage:
#   ./RUN_MakeSampleData_mac_Conda.command [RAW_DIR] [OUT_PROJECT] [WELLS...] 
# Defaults:
#   RAW_DIR=sample_raw, OUT_PROJECT=sample_project, WELLS="A01 A02 A03"
# Times and days can be overridden via env vars TIMES and DAYS, e.g.:
#   TIMES="00:00 12:00" DAYS="01" ./RUN_MakeSampleData_mac_Conda.command

ENV_NAME="organoid_roi_incucyte_imaging"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RAW_DIR="${1:-sample_raw}"
OUT_PROJ="${2:-sample_project}"

# Wells: pass as additional args or default list
shift $(( $# >= 2 ? 2 : $# )) || true
if [[ $# -gt 0 ]]; then
  WELLS=("$@")
else
  WELLS=(A01 A02 A03)
fi

# Allow overriding times/days via environment variables
TIMES_STR=${TIMES:-"00:00 12:00"}
DAYS_STR=${DAYS:-"01"}

echo "[make] Using env: $ENV_NAME"
echo "[make] RAW_DIR = $RAW_DIR"
echo "[make] OUT_PROJ = $OUT_PROJ"
echo "[make] WELLS   = ${WELLS[*]}"
echo "[make] DAYS    = $DAYS_STR"
echo "[make] TIMES   = $TIMES_STR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] Conda not found on PATH. Run 'conda init' and restart Terminal."
  exit 1
fi

echo "[run] Generating synthetic TIFFs into $RAW_DIR ..."
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tools/make_fake_data.py" \
  --raw "$RAW_DIR" \
  --wells ${WELLS[*]} \
  --days $DAYS_STR \
  --times $TIMES_STR

echo "[run] Reorganizing into $OUT_PROJ ..."
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/reorganize.py" \
  --raw "$RAW_DIR" \
  --out "$OUT_PROJ"

echo "[ok] Sample project ready at: $OUT_PROJ"
echo "    Try: conda run -n $ENV_NAME python $SCRIPT_DIR/gui_app.py"

