#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="organoid_roi_incucyte_imaging"
conda run -n "$ENV_NAME" python "$(dirname "$0")/gui_app.py"
