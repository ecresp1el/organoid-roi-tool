#!/usr/bin/env bash
set -euo pipefail
echo "=== Organoid ROI Tool: Conda setup & launch (v7) ==="
ENV_NAME="organoid_roi_incucyte_imaging"
if ! command -v conda >/dev/null 2>&1; then
  echo "[error] Conda not found on PATH. Run 'conda init' and restart Terminal."
  exit 1
fi
echo "[info] Using conda at: $(command -v conda)"
if ! conda env list | grep -q " $ENV_NAME[[:space:]]"; then
  echo "[info] Creating environment $ENV_NAME from environment.yml ..."
  conda env create -n "$ENV_NAME" -f "$(dirname "$0")/environment.yml"
else
  echo "[ok] Environment exists: $ENV_NAME"
fi
echo "[check] Verifying package imports in $ENV_NAME ..."
conda run -n "$ENV_NAME" python - <<'PY'
import sys
print("python", sys.version)
for m in ["napari","PySide6","numpy","tifffile","skimage","pandas","imagecodecs","pytest"]:
    __import__(m)
print("imports ok (including imagecodecs & pytest)")
PY
echo "[run] Launching GUI..."
conda run -n "$ENV_NAME" python "$(dirname "$0")/gui_app.py"
