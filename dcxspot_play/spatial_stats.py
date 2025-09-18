"""Spatial autocorrelation utilities (Moran's I) for ROI pixel data."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2-D array by (dy, dx) with zero fill at the borders."""

    out = np.zeros_like(arr)
    h, w = arr.shape
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    y0_dst = max(0, dy)
    y1_dst = y0_dst + (y1_src - y0_src)
    x0_dst = max(0, dx)
    x1_dst = x0_dst + (x1_src - x0_src)
    if y1_src > y0_src and x1_src > x0_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = arr[y0_src:y1_src, x0_src:x1_src]
    return out


def _neighbor_shifts(neighbors: int) -> Iterable[Tuple[int, int]]:
    if neighbors == 4:
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if neighbors == 8:
        return [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    raise ValueError("neighbors must be 4 or 8")


def morans_i_snapshot(image: np.ndarray, roi_mask: np.ndarray, neighbors: int = 8):
    """Compute global Moran's I and intermediates for local statistics."""

    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    mask = roi_mask.astype(bool, copy=False)
    n_pixels = int(mask.sum())
    if n_pixels < 5:
        raise ValueError("ROI has too few pixels for Moran's I (need >= 5)")

    x = image.astype(np.float64, copy=False)
    mean_roi = x[mask].mean()
    xc = np.zeros_like(x)
    xc[mask] = x[mask] - mean_roi

    neighbor_sum = np.zeros_like(xc)
    neighbor_count = np.zeros_like(xc)
    shifts = _neighbor_shifts(neighbors)

    for dy, dx in shifts:
        xc_shift = _shift(xc, dy, dx)
        mask_shift = _shift(mask.astype(np.float64), dy, dx) > 0.5
        neighbor_sum += xc_shift * mask_shift
        neighbor_count += mask_shift.astype(np.float64)

    numerator = float((xc * neighbor_sum * mask).sum())
    denominator = float((xc * xc * mask).sum()) + 1e-15
    w_sum = float((neighbor_count * mask).sum())
    I = (n_pixels / w_sum) * (numerator / denominator)
    return I, numerator, denominator, n_pixels, w_sum, xc, neighbor_sum, neighbor_count


def permutation_test_global(
    image: np.ndarray,
    roi_mask: np.ndarray,
    neighbors: int,
    permutations: int,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """Permutation test for global Moran's I (one-tailed positive autocorrelation)."""

    rng = np.random.default_rng(random_state)
    I_obs, *_ = morans_i_snapshot(image, roi_mask, neighbors=neighbors)

    vals = image[roi_mask].astype(np.float64, copy=False)
    I_perm = np.empty(permutations, dtype=np.float64)
    h, w = image.shape
    mask = roi_mask.astype(bool, copy=False)

    for k in range(permutations):
        perm = vals.copy()
        rng.shuffle(perm)
        img_perm = np.zeros((h, w), dtype=np.float64)
        img_perm[mask] = perm
        I_k, *_ = morans_i_snapshot(img_perm, mask, neighbors=neighbors)
        I_perm[k] = I_k

    mean_perm = float(I_perm.mean()) if permutations else float('nan')
    std_perm = float(I_perm.std(ddof=1)) if permutations > 1 else float('nan')
    p_pos = float((np.sum(I_perm >= I_obs) + 1) / (permutations + 1)) if permutations > 0 else float('nan')
    z = (I_obs - mean_perm) / (std_perm + 1e-15) if permutations > 1 and np.isfinite(std_perm) else float('nan')
    return {
        "I_observed": float(I_obs),
        "permutation_mean": mean_perm,
        "permutation_std": std_perm,
        "z_score": float(z),
        "p_value_one_tailed_positive": p_pos,
        "neighbors": int(neighbors),
        "permutations": int(permutations),
    }


def local_moran_map(xc: np.ndarray, roi_mask: np.ndarray, neighbor_sum_xc: np.ndarray) -> np.ndarray:
    """Local Moran map restricted to the ROI."""

    mask = roi_mask.astype(bool, copy=False)
    n_pixels = mask.sum()
    m2 = ((xc * xc * mask).sum() / (n_pixels + 1e-15)) + 1e-15
    local = np.zeros_like(xc, dtype=np.float64)
    local[mask] = (xc[mask] * neighbor_sum_xc[mask]) / m2
    return local


def local_moran_permutation_pvals(
    image: np.ndarray,
    roi_mask: np.ndarray,
    neighbors: int = 8,
    local_permutations: int = 199,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel permutation p-values for the Local Moran map (one-tailed)."""

    rng = np.random.default_rng(random_state)
    _, _, _, _, _, xc, neighbor_sum_xc, _ = morans_i_snapshot(image, roi_mask, neighbors=neighbors)
    Ii_obs = local_moran_map(xc, roi_mask, neighbor_sum_xc)

    vals = image[roi_mask].astype(np.float64, copy=False)
    h, w = image.shape
    mask = roi_mask.astype(bool, copy=False)
    perm_exceed = np.zeros((h, w), dtype=np.float64)

    for _ in range(local_permutations):
        perm = vals.copy()
        rng.shuffle(perm)
        img_perm = np.zeros((h, w), dtype=np.float64)
        img_perm[mask] = perm
        _, _, _, _, _, xc_p, neighbor_sum_xc_p, _ = morans_i_snapshot(img_perm, mask, neighbors=neighbors)
        Ii_k = local_moran_map(xc_p, mask, neighbor_sum_xc_p)
        perm_exceed += (Ii_k >= Ii_obs).astype(np.float64)

    p_map = np.zeros_like(image, dtype=np.float64)
    p_map[mask] = (perm_exceed[mask] + 1.0) / (local_permutations + 1.0)
    return Ii_obs, p_map

