#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="organoid_roi_incucyte_imaging"

# Optional CLI usage: RUN_Reorganize_mac_Conda_v2.command /path/to/raw /path/to/out [min_col] [rows]
RAW="${1:-}"
OUT="${2:-}"
MINCOL="${3:-}"
ROWS="${4:-}"

if [ -z "${RAW}" ]; then
  read -p "Enter path to your raw_images folder: " RAW
fi
if [ -z "${OUT}" ]; then
  read -p "Enter desired output project folder: " OUT
fi
if [ -z "${MINCOL}" ]; then
  read -p "Only include wells with column >= (default 1; e.g., 4): " MINCOL || true
fi
if [ -z "${ROWS}" ]; then
  read -p "Only include row letters (default ABCDEFGH): " ROWS || true
fi

MINCOL="${MINCOL:-1}"
ROWS="${ROWS:-ABCDEFGH}"

echo "[run] rows=${ROWS}  min_col>=${MINCOL}"
conda run -n "$ENV_NAME" python "$(dirname "$0")/reorganize.py" --raw "$RAW" --out "$OUT" --min_col "$MINCOL" --rows "$ROWS"