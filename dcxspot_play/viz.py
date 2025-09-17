"""Visualization helpers for DCX Otsu segmentation outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from skimage import segmentation


def _percentile_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Return a float image scaled to [0, 1] using robust percentiles (ignoring NaNs)."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    stretched = (img - vmin) / (vmax - vmin)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def _show_masked(ax, img: np.ndarray, title: str, cmap: str = "gray") -> None:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=(0, 0, 0, 0))
    display = _percentile_stretch(img)
    im = ax.imshow(display, cmap=cmap_obj, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    return im


def _overlay_canvas_setup() -> plt.Axes:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.axis("off")
    return ax


def save_overlay_ids(
    mcherry_masked: np.ndarray,
    labels: np.ndarray,
    out_png: str | Path,
    n_final: int,
    otsu_thr: float,
) -> np.ndarray:
    """Save an ID overlay PNG and return the rendered RGB array.

    Parameters
    ----------
    mcherry_masked:
        ROI-masked mCherry image (NaN outside ROI) for visualisation.
    labels:
        Labeled clusters (int32). 0 denotes background.
    out_png:
        Destination path for the saved overlay figure.
    n_final:
        Number of clusters remaining after filtering.
    otsu_thr:
        Threshold used during segmentation; included in the legend.

    Returns
    -------
    np.ndarray
        RGB image array extracted from the saved figure for reuse in panel layouts.
    """
    if mcherry_masked.shape != labels.shape:
        raise ValueError("mCherry image and labels must have the same shape for overlay")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    stretched = _percentile_stretch(mcherry_masked)
    ax.imshow(stretched, cmap=cmap, interpolation="nearest")

    boundaries = segmentation.find_boundaries(labels, mode="inner")
    if np.any(boundaries):
        ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="autumn", alpha=0.9)

    for cluster_id in np.unique(labels):
        if cluster_id == 0:
            continue
        y, x = np.argwhere(labels == cluster_id).mean(axis=0)
        ax.text(
            x,
            y,
            str(int(cluster_id)),
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
        )

    legend_text = f"N clusters: {n_final}\notsu = {otsu_thr:.4g}"
    ax.text(
        0.02,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
        va="bottom",
        ha="left",
    )

    ax.set_title("DCX clusters (IDs)")
    ax.axis("off")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    overlay_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_rgb = overlay_rgb.reshape((height, width, 3))
    plt.close(fig)
    return overlay_rgb


def _ensure_rgb(img) -> np.ndarray:
    if isinstance(img, (str, Path)):
        return plt.imread(img)
    return np.asarray(img)


def save_panel_1x3(
    bf_masked: np.ndarray,
    mcherry_masked: np.ndarray,
    overlay_img,
    out_png: str | Path,
) -> None:
    """Save a 1×3 QC panel with BF, mCherry, and cluster overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=300)

    _show_masked(axes[0], bf_masked, "BF (masked)")
    _show_masked(axes[1], mcherry_masked, "mCherry (masked)")

    axes[2].imshow(_ensure_rgb(overlay_img))
    axes[2].set_title("DCX clusters (IDs)")
    axes[2].axis("off")

    fig.tight_layout(w_pad=1.0)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_panel_1x4(
    bf_masked: np.ndarray,
    mcherry_masked: np.ndarray,
    dcx_binary: np.ndarray,
    overlay_img,
    out_png: str | Path,
) -> None:
    """Save a 1×4 QC panel that also includes the binary DCX mask."""
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3), dpi=300)

    _show_masked(axes[0], bf_masked, "BF (masked)")
    _show_masked(axes[1], mcherry_masked, "mCherry (masked)")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    binary = np.array(dcx_binary, dtype=float)
    binary_display = np.where(np.isfinite(binary), binary, np.nan)
    axes[2].imshow(binary_display, cmap=cmap, interpolation="nearest")
    axes[2].set_title("DCX binary (masked)")
    axes[2].axis("off")

    axes[3].imshow(_ensure_rgb(overlay_img))
    axes[3].set_title("DCX clusters (IDs)")
    axes[3].axis("off")

    fig.tight_layout(w_pad=1.0)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
