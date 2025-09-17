"""Otsu-based DCX spot-finding pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import watershed

from .quantify import measure_from_raw
from .utils import (
    apply_roi_mask,
    read_image,
    read_mask,
    save_csv,
    save_json,
    save_tiff,
)
from .viz import save_overlay_ids, save_panel_1x3, save_panel_1x4


DEFAULT_PARAMS = {
    "min_area": 24,
    "max_area": 8000,
    "min_distance": 3,
    "morph_radius": 0,
    "normalize_percentiles": (0.0, 0.0),
    "save_1x4": False,
    "output_dir": None,
    "image_id": None,
}


def _normalize_for_segmentation(data: np.ndarray, roi: np.ndarray, percentiles) -> np.ndarray:
    lo, hi = percentiles
    if lo == 0 and hi == 0:
        return data.astype(np.float32)
    roi_vals = data[roi]
    if roi_vals.size == 0:
        return data.astype(np.float32)
    lo_val, hi_val = np.percentile(roi_vals, [lo, hi])
    if hi_val <= lo_val:
        return data.astype(np.float32)
    seg = (data.astype(np.float32) - lo_val) / (hi_val - lo_val)
    return np.clip(seg, 0.0, 1.0)


def _local_maxima(distance: np.ndarray, mask: np.ndarray, min_distance: int) -> np.ndarray:
    kwargs = dict(labels=mask, min_distance=max(1, int(min_distance)), exclude_border=False)
    local_max = peak_local_max(distance, **kwargs)
    if local_max.size == 0:
        markers, _ = ndi.label(mask)
        return markers
    if local_max.ndim == 2:
        markers = np.zeros_like(distance, dtype=np.int32)
        for idx, (r, c) in enumerate(local_max, start=1):
            markers[r, c] = idx
        if markers.max() == 0:
            markers, _ = ndi.label(mask)
        return markers
    markers, _ = ndi.label(local_max)
    if markers.max() == 0:
        markers, _ = ndi.label(mask)
    return markers


def _filter_labels(labels: np.ndarray, min_area: int, max_area: int) -> Dict[str, Any]:
    flat = labels.ravel()
    areas = np.bincount(flat)
    keep_ids = []
    removed_small = 0
    removed_large = 0
    for label_id in range(1, len(areas)):
        area = areas[label_id]
        if area == 0:
            continue
        if min_area and area < min_area:
            removed_small += 1
            continue
        if max_area and area > max_area:
            removed_large += 1
            continue
        keep_ids.append(label_id)

    mapping = {old: new for new, old in enumerate(keep_ids, start=1)}
    filtered = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        filtered[labels == old] = new
    return {
        "labels": filtered,
        "keep_ids": keep_ids,
        "removed_small": removed_small,
        "removed_large": removed_large,
        "n_candidates": int(labels.max()),
    }


def run_one(
    mcherry_path: str | Path,
    roi_path: str | Path,
    bf_path: str | Path | None = None,
    params: dict | None = None,
) -> dict:
    """Run the DCX Otsu pipeline on a single image and write all required artefacts."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    mcherry_path = Path(mcherry_path)
    roi_path = Path(roi_path)
    bf_path = Path(bf_path) if bf_path is not None else None

    mcherry_raw = read_image(mcherry_path)
    if mcherry_raw.ndim != 2:
        raise ValueError("mCherry image must be 2-D")
    roi = read_mask(roi_path)
    if roi.shape != mcherry_raw.shape:
        raise ValueError("ROI mask and mCherry image must have identical shapes")

    bf_raw = None
    if bf_path is not None:
        bf_raw = read_image(bf_path)
        if bf_raw.shape != mcherry_raw.shape:
            raise ValueError("BF image must have the same shape as mCherry image")

    roi_bool = roi.astype(bool)
    if roi_bool.sum() == 0:
        otsu_thr = float("nan")
        bw = np.zeros_like(roi_bool, dtype=bool)
        labels = np.zeros_like(roi_bool, dtype=np.int32)
        filtered_info = {
            "labels": labels,
            "keep_ids": [],
            "removed_small": 0,
            "removed_large": 0,
            "n_candidates": 0,
        }
    else:
        seg_img = _normalize_for_segmentation(
            mcherry_raw.astype(np.float32),
            roi_bool,
            params["normalize_percentiles"],
        )
        otsu_thr = float(threshold_otsu(seg_img[roi_bool]))
        bw = (seg_img > otsu_thr) & roi_bool

        morph_radius = int(params["morph_radius"] or 0)
        if morph_radius > 0:
            selem = disk(morph_radius)
            bw = binary_opening(bw, selem)
            bw = binary_closing(bw, selem)

        if not bw.any():
            labels = np.zeros_like(bw, dtype=np.int32)
            filtered_info = {
                "labels": labels,
                "keep_ids": [],
                "removed_small": 0,
                "removed_large": 0,
                "n_candidates": 0,
            }
        else:
            distance = ndi.distance_transform_edt(bw)
            markers = _local_maxima(distance, bw, params["min_distance"])
            labels_raw = watershed(-distance, markers, mask=bw)
            filtered_info = _filter_labels(
                labels_raw,
                min_area=int(params["min_area"]),
                max_area=int(params["max_area"]),
            )
            labels = filtered_info["labels"]
            bw = labels > 0

    n_final = int(labels.max())

    image_id = params.get("image_id") or mcherry_path.stem
    output_dir = Path(params.get("output_dir") or mcherry_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Masked images
    mcherry_masked = apply_roi_mask(mcherry_raw, roi_bool, outside="nan")
    mcherry_masked_zero = apply_roi_mask(mcherry_raw, roi_bool, outside="zero")

    if bf_raw is not None:
        bf_masked = apply_roi_mask(bf_raw, roi_bool, outside="nan")
        bf_masked_zero = apply_roi_mask(bf_raw, roi_bool, outside="zero")
    else:
        bf_masked = np.full_like(mcherry_masked, np.nan)
        bf_masked_zero = np.zeros_like(mcherry_masked, dtype=np.float32)

    dcx_binary = bw.astype(np.uint8)
    dcx_binary_masked = apply_roi_mask(dcx_binary.astype(np.float32), roi_bool, outside="nan")

    roi_vals = mcherry_raw[roi_bool]
    if roi_vals.size > 0:
        roi_mean = float(np.mean(roi_vals))
        roi_median = float(np.median(roi_vals))
        roi_std = float(np.std(roi_vals))
        roi_min = float(np.min(roi_vals))
        roi_max = float(np.max(roi_vals))
    else:
        roi_mean = roi_median = roi_std = roi_min = roi_max = float("nan")

    # Quantification
    df = measure_from_raw(mcherry_raw, labels)
    df["image_id"] = image_id
    df["otsu_threshold"] = otsu_thr
    df["min_area_px"] = params["min_area"]
    df["max_area_px"] = params["max_area"]
    df["min_distance_px"] = params["min_distance"]

    qc = {
        "image_id": image_id,
        "n_candidates_before_filters": filtered_info["n_candidates"],
        "removed_small": filtered_info["removed_small"],
        "removed_large": filtered_info["removed_large"],
        "n_final": n_final,
        "otsu_threshold": otsu_thr,
        "roi_mean_intensity": roi_mean,
        "roi_median_intensity": roi_median,
        "roi_std_intensity": roi_std,
        "roi_min_intensity": roi_min,
        "roi_max_intensity": roi_max,
        "min_area_px": params["min_area"],
        "max_area_px": params["max_area"],
        "min_distance_px": params["min_distance"],
        "morph_radius_px": params["morph_radius"],
        "normalize_percentiles": list(params["normalize_percentiles"]),
    }

    # Save artefacts
    save_tiff(output_dir / f"{image_id}_labels.tif", labels.astype(np.int32))
    save_tiff(output_dir / f"{image_id}_dcx_binary.tif", dcx_binary.astype(np.uint8))

    save_tiff(output_dir / f"{image_id}_bf_masked.tif", bf_masked.astype(np.float32))
    save_tiff(output_dir / f"{image_id}_bf_masked_zero.tif", bf_masked_zero.astype(np.float32))
    save_tiff(output_dir / f"{image_id}_mcherry_masked.tif", mcherry_masked.astype(np.float32))
    save_tiff(output_dir / f"{image_id}_mcherry_masked_zero.tif", mcherry_masked_zero.astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    raw_display = np.nan_to_num(mcherry_masked, nan=0.0)
    axes[0].imshow(raw_display, cmap="gray")
    axes[0].set_title("Raw (ROI masked)")
    axes[0].axis("off")
    axes[1].imshow(raw_display, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~bw, bw), cmap="cool", alpha=0.4)
    axes[1].set_title("Raw + DCX mask")
    axes[1].axis("off")
    fig.tight_layout()
    debug_png = output_dir / f"{image_id}_debug_mask_overlay.png"
    fig.savefig(debug_png, bbox_inches="tight")
    plt.close(fig)

    save_csv(output_dir / f"{image_id}_spots.csv", df)
    save_json(output_dir / f"{image_id}_qc.json", qc)

    overlay_png = output_dir / f"{image_id}_overlay_ids.png"
    overlay_rgb = save_overlay_ids(mcherry_masked, labels, overlay_png, n_final, otsu_thr)

    panel_1x3_path = output_dir / f"{image_id}_panel_1x3.png"
    save_panel_1x3(bf_masked, mcherry_masked, overlay_rgb, panel_1x3_path)

    if params.get("save_1x4"):
        panel_1x4_path = output_dir / f"{image_id}_panel_1x4.png"
        save_panel_1x4(bf_masked, mcherry_masked, dcx_binary_masked, overlay_rgb, panel_1x4_path)

    result = {
        "labels": labels,
        "bw": bw.astype(bool),
        "df": df,
        "qc": qc,
        "overlay_rgb": overlay_rgb,
        "panel_1x3_path": str(panel_1x3_path),
    }
    if params.get("save_1x4"):
        result["panel_1x4_path"] = str(panel_1x4_path)
    return result
