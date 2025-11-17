"""Utility helpers for loading optional organoid masks.

The analyses can restrict their per-pixel statistics to the organoid interior
by loading a binary mask saved next to the projection TIFFs. Masks are expected
to live inside each run folder's ``16bit/`` directory and follow the naming
pattern ``mask*.tif(f)``. Only a single mask should exist per run: when more
than one candidate is present the helper raises ``ValueError`` so the user can
clean up the folder before rerunning the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff  # type: ignore

MASK_GLOB_PATTERNS = (
    "mask.tif",
    "mask.tiff",
    "mask_*.tif",
    "mask_*.tiff",
)


def discover_mask_path(run_folder: Path) -> Optional[Path]:
    """Return the unique mask path for ``run_folder`` or ``None`` if missing."""

    candidate_dir = run_folder / "16bit"
    if not candidate_dir.exists():
        return None

    candidates = []
    for pattern in MASK_GLOB_PATTERNS:
        candidates.extend(candidate_dir.glob(pattern))

    unique_candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not unique_candidates:
        return None
    if len(unique_candidates) > 1:
        formatted = "\n".join(str(path) for path in unique_candidates)
        raise ValueError(
            f"Multiple mask files detected for {run_folder}:\n{formatted}\n"
            "Please keep only one mask per run."
        )
    return unique_candidates[0]


def load_mask_array(run_folder: Path) -> Optional[np.ndarray]:
    """Read and return the binary mask array for ``run_folder`` if it exists."""

    mask_path = discover_mask_path(run_folder)
    if mask_path is None:
        return None
    mask = tiff.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return mask.astype(bool, copy=False)
