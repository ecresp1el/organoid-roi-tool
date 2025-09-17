"""Utility helpers for loading, masking, and saving data in the DCX Otsu pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - runtime validation
    raise SystemExit(
        "tifffile is required. Activate the organoid ROI environment or install it."
    ) from exc

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional fallback
    Image = None


def _read_tiff(path: Path) -> np.ndarray:
    """Load a TIFF into a numpy array, falling back to Pillow if imagecodecs is missing."""
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if "imagecodecs" in str(exc) and Image is not None:
            with Image.open(path) as img:
                return np.array(img)
        raise


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file (TIFF preferred) into a numpy array."""
    arr = _read_tiff(Path(path))
    return np.asarray(arr)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask image and return it as a boolean array (True inside ROI)."""
    mask = read_image(path)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D; got shape {mask.shape} from {path}")
    return mask.astype(bool)


def apply_roi_mask(
    img: np.ndarray,
    roi: np.ndarray,
    *,
    outside: Literal["nan", "zero"] = "nan",
) -> np.ndarray:
    """Apply ROI to an image, replacing pixels outside ROI with NaN or 0."""
    if img.shape[-2:] != roi.shape:
        raise ValueError(
            f"Image spatial shape {img.shape[-2:]} does not match ROI shape {roi.shape}"
        )
    roi_bool = roi.astype(bool)
    out = img.astype(np.float32, copy=True)
    if out.ndim == 2:
        if outside == "nan":
            out[~roi_bool] = np.nan
        else:
            out[~roi_bool] = 0.0
        return out

    # Broadcast the ROI mask across leading axes (channels, z, etc.).
    expand_shape = (1,) * (out.ndim - 2) + roi_bool.shape
    roi_expanded = roi_bool.reshape(expand_shape)
    if outside == "nan":
        out = out.copy()
        out[~roi_expanded] = np.nan
    else:
        out[~roi_expanded] = 0.0
    return out


def save_tiff(path: str | Path, arr: np.ndarray, dtype=None) -> None:
    """Persist an array to disk as a TIFF file."""
    if dtype is not None:
        arr = arr.astype(dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    """Persist a DataFrame to CSV with UTF-8 encoding and no index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(path: str | Path, obj) -> None:
    """Persist a JSON-serialisable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
