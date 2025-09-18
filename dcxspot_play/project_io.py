"""Helpers for locating and loading project image assets for ROI analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import read_image, read_mask


def resolve_brightfield_path(row: pd.Series, project_root: Path) -> Path:
    """Return the absolute brightfield image path for a measurement row.

    The row is expected to originate from ``roi_measurements.csv`` and include
    ``image_relpath`` (portable) and ``image_path`` (legacy absolute). The
    portable path is preferred because it keeps projects movable across
    machines/drives.
    """

    project_root = project_root.expanduser().resolve()

    rel = row.get("image_relpath")
    if isinstance(rel, str) and rel:
        candidate = (project_root / rel).expanduser().resolve()
        if candidate.exists():
            return candidate

    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path:
        candidate = Path(image_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate image for row with relpath={rel!r} and image_path={row.get('image_path')!r}"
    )


def locate_roi_artifacts(row: pd.Series, project_root: Path) -> Tuple[Path, Path, Path]:
    """Return brightfield, mask, and fluorescence paths for a measurement row."""

    brightfield_path = resolve_brightfield_path(row, project_root)
    mask_path = brightfield_path.with_name(f"{brightfield_path.stem}_mask.tif")
    fluor_path = brightfield_path.parent / "fluorescence" / f"{brightfield_path.stem}_mcherry.tif"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing ROI mask for {brightfield_path}")
    if not fluor_path.exists():
        raise FileNotFoundError(f"Missing fluorescence image for {brightfield_path}")

    return brightfield_path, mask_path, fluor_path


def load_roi_images(row: pd.Series, project_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load brightfield, mask (bool), and fluorescence arrays for a row."""

    bf_path, mask_path, fluor_path = locate_roi_artifacts(row, project_root)
    brightfield = read_image(bf_path).astype(np.float32, copy=False)
    mask = read_mask(mask_path).astype(bool, copy=False)
    fluorescence = read_image(fluor_path).astype(np.float32, copy=False)
    return brightfield, mask, fluorescence

