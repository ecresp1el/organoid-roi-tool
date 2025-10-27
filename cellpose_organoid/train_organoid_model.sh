#!/usr/bin/env bash
# Train a Cellpose model that segments the ENTIRE organoid as ONE ROI.
# Uses your existing WT/KO folders as the training set.
# Steps (printed below as [1/4] ... [4/4]):
#   1. Link the TIFFs referenced in the metadata CSV into a temporary workspace
#   2. Auto-generate Cellpose *_seg.npy files if they do not exist yet
#   3. Train a custom Cellpose model and log the run
#   4. (Optional) Re-use the new model for a quick smoke-test

set -euo pipefail

# ---- DATASET CONFIG (EDIT AS NEEDED) ----
PROJECT_ROOT="/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder"
ANALYSIS="PCDHvsLHX6_WTvsKO_IHC"
PROJECTION_TYPES=("max")
EXPERIMENT_GROUPS=("WT" "KO")
# For single-channel metadata you can narrow the export to specific channel_slugs, e.g.
# CHANNEL_SLUGS=("LHX6" "PCDH19" "DAPI_reference")
CHANNEL_SLUGS=()
MAKE_SEG_VERBOSE="true"   # set to "false" if you want quieter Cellpose output

# Location of helper scripts (stays inside the repo)
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cellpose_organoid"

# Workspace kept alongside the data (outside the git repo)
WORKSPACE_ROOT="${PROJECT_ROOT}/cellpose_organoid_workspace"
TRAIN_DIR="${WORKSPACE_ROOT}/data/train"
MODEL_DIR="${WORKSPACE_ROOT}/models"
LOG_DIR="${WORKSPACE_ROOT}/logs"

# Metadata CSV emitted by prepare_for_cellprofiler_cellpose.py
METADATA_MULTI="${PROJECT_ROOT}/cellpose_multichannel_zcyx/cellpose_multichannel_metadata.csv"
METADATA_SINGLE="${PROJECT_ROOT}/cellprofilerandcellpose_metadata.csv"
if [[ -f "${METADATA_MULTI}" ]]; then
  METADATA_CSV="${METADATA_MULTI}"
  echo "[INFO] Using multi-channel metadata: ${METADATA_CSV}"
else
  METADATA_CSV="${METADATA_SINGLE}"
  echo "[WARN] Multi-channel metadata not found; falling back to ${METADATA_CSV}"
fi

if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "[ERROR] No metadata CSV found at ${METADATA_MULTI} or ${METADATA_SINGLE}" >&2
  exit 1
fi

# Cellpose params tuned for "whole organoid"
MODEL_INIT="cyto3"        # initializer
DIAMETER=1500             # ~whole organoid width in px
CHAN=0                    # 0=gray
CHAN2=0                   # 0=none
FLOW_THR=0.1
CELLP_THR=-6.0
EPOCHS=500

# Ensure the workspace folders exist before we start writing anything.
mkdir -p "${TRAIN_DIR}" "${MODEL_DIR}" "${LOG_DIR}"

echo "[1/4] Preparing training workspace via metadata"
SYMLINK_SCRIPT="${SCRIPT_ROOT}/scripts/prepare_training_from_metadata.py"

PROJECTION_ARGS=()
if [[ -n "${PROJECTION_TYPES+set}" ]]; then
  for proj in "${PROJECTION_TYPES[@]}"; do
    PROJECTION_ARGS+=("--projection" "${proj}")
  done
fi

GROUP_ARGS=()
if [[ -n "${EXPERIMENT_GROUPS+set}" ]]; then
  for grp in "${EXPERIMENT_GROUPS[@]}"; do
    GROUP_ARGS+=("--group" "${grp}")
  done
fi

ANALYSIS_ARGS=()
if [[ -n "${ANALYSIS}" ]]; then
  ANALYSIS_ARGS+=("--analysis" "${ANALYSIS}")
fi

CHANNEL_ARGS=()
if [[ -n "${CHANNEL_SLUGS+set}" && ${#CHANNEL_SLUGS[@]} -gt 0 ]]; then
  for slug in "${CHANNEL_SLUGS[@]}"; do
    CHANNEL_ARGS+=("--channel-slug" "${slug}")
  done
fi

# Build the Python command as an array so optional filters can be appended safely.
SYMLINK_CMD=(
  python "${SYMLINK_SCRIPT}"
  --metadata "${METADATA_CSV}"
  --output "${TRAIN_DIR}"
  --clear-output
)

if ((${#ANALYSIS_ARGS[@]})); then
  SYMLINK_CMD+=("${ANALYSIS_ARGS[@]}")
fi
if ((${#PROJECTION_ARGS[@]})); then
  SYMLINK_CMD+=("${PROJECTION_ARGS[@]}")
fi
if ((${#GROUP_ARGS[@]})); then
  SYMLINK_CMD+=("${GROUP_ARGS[@]}")
fi
if ((${#CHANNEL_ARGS[@]})); then
  SYMLINK_CMD+=("${CHANNEL_ARGS[@]}")
fi

"${SYMLINK_CMD[@]}"

if ! compgen -G "${TRAIN_DIR}/*.tif" >/dev/null; then
  echo "[ERROR] No TIFFs found in ${TRAIN_DIR} after metadata linking. Check the filters above." >&2
  exit 1
fi

echo "[2/4] Auto-labeling (creating *_seg.npy) where missing"
# Generates Cellpose labels for any TIFF in TRAIN_DIR that does not already have a *_seg.npy file.
# If you already saved masks manually via the Cellpose GUI, those *_seg.npy files
# live alongside the TIFFs and this step immediately skips them.
MAKE_SEG_CMD=(
  python "${SCRIPT_ROOT}/scripts/make_seg_from_model.py"
  --dirs "${TRAIN_DIR}"
  --model "${MODEL_INIT}"
  --diameter "${DIAMETER}"
  --chan "${CHAN}" --chan2 "${CHAN2}"
  --flow_threshold "${FLOW_THR}"
  --cellprob_threshold "${CELLP_THR}"
)
if [[ "${MAKE_SEG_VERBOSE}" == "true" ]]; then
  MAKE_SEG_CMD+=(--verbose)
fi

"${MAKE_SEG_CMD[@]}"

echo "[3/4] Training custom model on ${TRAIN_DIR}"
# NOTE: Cellpose saves checkpoints into TRAIN_DIR/models/ by default, so we tee the log separately.
python -m cellpose \
  --train \
  --dir "${TRAIN_DIR}" \
  --mask_filter _seg.npy \
  --pretrained_model "${MODEL_INIT}" \
  --chan "${CHAN}" --chan2 "${CHAN2}" \
  --learning_rate 0.2 \
  --min_train_masks 1 \
  --save_every 50 \
  --n_epochs "${EPOCHS}" | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# Trained weights land in ${TRAIN_DIR}/models/
# Optionally copy the latest model to a stable location inside the workspace root:
LATEST_MODEL_DIR="$(ls -td "${TRAIN_DIR}"/models/* | head -n1 || true)"
if [[ -n "${LATEST_MODEL_DIR}" ]]; then
  cp -R "${LATEST_MODEL_DIR}" "${MODEL_DIR}/organoid_roi_$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] Copied trained model to ${MODEL_DIR}"
fi

echo "[4/4] Quick smoke-test: re-run the trained model on WT & KO (writes *_seg.npy if improved)"
# If you want to use the newly trained model explicitly, uncomment below and set MODEL_PATH:
# MODEL_PATH="${MODEL_DIR}/organoid_roi_YYYYMMDD_HHMMSS"   # <- fill with the folder printed above
# python "${SCRIPT_ROOT}/scripts/make_seg_from_model.py" \
#   --dirs "${TRAIN_DIR}" \
#   --model "${MODEL_PATH}" \
#   --diameter "${DIAMETER}" \
#   --chan "${CHAN}" --chan2 "${CHAN2}" \
#   --flow_threshold "${FLOW_THR}" \
#   --cellprob_threshold "${CELLP_THR}"

echo "[DONE] Training complete."
