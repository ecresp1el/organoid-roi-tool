#!/usr/bin/env bash
# Train a Cellpose model that segments the ENTIRE organoid as ONE ROI.
# Uses your existing WT/KO folders as the training set.

set -euo pipefail

# ---- EDIT THESE PATHS IF NEEDED ----
BASE="/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max"
WT_DIR="${BASE}/WT"
KO_DIR="${BASE}/KO"

# repo-relative workspace (safe to keep inside your repo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${REPO_ROOT}/cellpose_organoid"
TRAIN_DIR="${WORK}/data/train"
MODEL_DIR="${WORK}/models"
LOG_DIR="${WORK}/logs"

# Cellpose params tuned for "whole organoid"
MODEL_INIT="cyto3"        # initializer
DIAMETER=1500             # ~whole organoid width in px
CHAN=0                    # 0=gray
CHAN2=0                   # 0=none
FLOW_THR=0.1
CELLP_THR=-6.0
EPOCHS=500

mkdir -p "${TRAIN_DIR}" "${MODEL_DIR}" "${LOG_DIR}"

echo "[1/4] Symlinking WT+KO images into ${TRAIN_DIR}"
find "${WT_DIR}" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -exec ln -sf {} "${TRAIN_DIR}/" \;
find "${KO_DIR}" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -exec ln -sf {} "${TRAIN_DIR}/" \;

echo "[2/4] Auto-labeling (creating *_seg.npy) where missing"
python "${WORK}/scripts/make_seg_from_model.py" \
  --dirs "${TRAIN_DIR}" \
  --model "${MODEL_INIT}" \
  --diameter "${DIAMETER}" \
  --chan "${CHAN}" --chan2 "${CHAN2}" \
  --flow_threshold "${FLOW_THR}" \
  --cellprob_threshold "${CELLP_THR}"

echo "[3/4] Training custom model on ${TRAIN_DIR}"
# NOTE: Cellpose saves checkpoints into train/models/ by default.
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
# Optionally copy the latest model to a stable location:
LATEST_MODEL_DIR="$(ls -td "${TRAIN_DIR}"/models/* | head -n1 || true)"
if [[ -n "${LATEST_MODEL_DIR}" ]]; then
  cp -R "${LATEST_MODEL_DIR}" "${MODEL_DIR}/organoid_roi_$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] Copied trained model to ${MODEL_DIR}"
fi

echo "[4/4] Quick smoke-test: re-run the trained model on WT & KO (writes *_seg.npy if improved)"
# If you want to use the newly trained model explicitly, uncomment below and set MODEL_PATH:
# MODEL_PATH="${MODEL_DIR}/organoid_roi_YYYYMMDD_HHMMSS"   # <- fill with the folder printed above
# python "${WORK}/scripts/make_seg_from_model.py" --dirs "${WT_DIR}" "${KO_DIR}" --model "${MODEL_PATH}" --diameter "${DIAMETER}" --chan "${CHAN}" --chan2 "${CHAN2}" --flow_threshold "${FLOW_THR}" --cellprob_threshold "${CELLP_THR}"

echo "[DONE] Training complete."
