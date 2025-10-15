"""
Simple Projection Export Tool
==============================

Purpose
-------
Create a reproducible, easy-to-read snapshot of each channel contained in an
Imaris ``.ims`` file by exporting the core statistical projections (max, mean,
median) and accompanying visual guides. The script is designed for quick
quality-control passes and downstream quantification without modifying the raw
data.

Inputs
------
- A folder containing stitched Imaris ``.ims`` files. (Sidecar ``.txt`` files
  are ignored.)
- Optional ``--recursive`` flag to recurse into sub-folders.

Outputs (per ``.ims`` file)
---------------------------
``simple_projections/<ims_stem>/16bit/``
    Max / mean / median projections scaled to unsigned 16-bit. If the source
    data already fits in ≤16-bit, the original dtype is preserved.

``simple_projections/<ims_stem>/8bit/``
    Max / mean / median projections normalised to unsigned 8-bit for quick
    inspection by tools that expect 8-bit imagery.

``simple_projections/<ims_stem>/figures/``
    Colourised PNG panels (one per projection) grouped into three sub-folders:

    * ``raw_min_max/`` – raw min/max stretch
    * ``percentile_95/`` – min → 95th percentile stretch
    * ``median_mad/`` – median ± 3×MAD stretch

    A ``README.txt`` inside ``figures/`` describes each scaling strategy and
    notes that underlying TIFF exports are untouched.

Processing Steps
----------------
1. Discover matching ``.ims`` files.
2. Load channel metadata via :func:`imaris_tools.metadata.read_metadata` to
   retrieve channel names and colours.
3. Stream the HDF5 datasets with ``h5py`` and compute raw max/mean/median
   projections (no filtering or background subtraction).
4. Save a 16-bit and an 8-bit TIFF per projection.
5. Generate three diagnostic PNG panels per projection using channel-specific
   pseudocolours and clearly labelled colour bars.
6. Emit a figure manifest so collaborators can understand the scaling without
   reading the source.

Dependencies
------------
- ``imaris_tools`` package in this repository (accessed via the local symlink).
- ``h5py``, ``numpy``, ``tifffile``, ``matplotlib`` (installed through the
  project's Conda environment).

Usage Example
-------------
.. code-block:: console

    conda activate organoid_roi_incucyte_imaging
    cd path/to/organoid-roi-tool
    python simple_projection_tool/simple_channel_projections.py \\
        --source /path/to/ims_folder

The script may be invoked from any directory, but it expects to be run after
activating the same environment that powers the rest of this project so that
``imaris_tools`` and its dependencies are importable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py  # type: ignore
import numpy as np
import tifffile as tiff  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from imaris_tools.metadata import read_metadata, default_channel_color

DEFAULT_SOURCE = Path("/Volumes/Manny4TBUM/2025-10-15")
OUTPUT_FOLDER_NAME = "simple_projections"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save max/mean/median channel projections for Imaris .ims files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing .ims files (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .ims files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        print(f"[error] Source directory {source} does not exist or is not a folder.")
        return 1

    ims_files = list(_collect_files(source, recursive=args.recursive))
    if not ims_files:
        print(f"[warn] No .ims files found in {source}")
        return 0

    print(f"[info] Found {len(ims_files)} .ims file(s) in {source}")
    for index, ims_path in enumerate(ims_files, start=1):
        print(f"[step] ({index}/{len(ims_files)}) Processing {ims_path.name}")
        try:
            _process_file(ims_path)
        except Exception as exc:
            print(f"[error] Failed to process {ims_path}: {exc}")
    print("[done] Finished exporting projections.")
    return 0


def _collect_files(folder: Path, *, recursive: bool) -> Iterable[Path]:
    if recursive:
        iterator = folder.rglob("*.ims")
    else:
        iterator = folder.glob("*.ims")
    return sorted(path for path in iterator if path.is_file())


def _process_file(path: Path) -> None:
    metadata = read_metadata(path)
    channel_lookup = {channel.index: channel for channel in metadata.channels}

    output_root = path.parent / OUTPUT_FOLDER_NAME / path.stem
    output_16 = output_root / "16bit"
    output_8 = output_root / "8bit"
    output_fig = output_root / "figures"
    output_16.mkdir(parents=True, exist_ok=True)
    output_8.mkdir(parents=True, exist_ok=True)
    output_fig.mkdir(parents=True, exist_ok=True)

    figure_directories = {
        "raw": output_fig / "raw_min_max",
        "p95": output_fig / "percentile_95",
        "mad": output_fig / "median_mad",
    }
    for directory in figure_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as handle:
        channel_paths = _discover_channel_paths(handle)
        if not channel_paths:
            print(f"[warn] No channels found in {path}")
            return

        for channel_index, dataset in channel_paths:
            data = dataset[()]  # NumPy array (Z, Y, X)
            if data.ndim < 3:
                print(f"[warn] Unexpected channel shape {data.shape} in {path}")
                continue

            max_proj = np.max(data, axis=0)
            mean_proj = np.mean(data, axis=0)
            median_proj = np.median(data, axis=0)

            channel_info = channel_lookup.get(channel_index)
            channel_name = channel_info.name if channel_info is not None else f"Channel_{channel_index:02d}"
            sanitized = _sanitize_name(channel_name)
            channel_color = channel_info.color_rgb if channel_info is not None else default_channel_color(channel_index)

            projections = [
                ("max", max_proj),
                ("mean", mean_proj),
                ("median", median_proj),
            ]

            for label, array in projections:
                raw_path = output_16 / f"{sanitized}_{label}.tif"
                eight_path = output_8 / f"{sanitized}_{label}.tif"

                arr16 = _to_uint16(array)
                arr8 = _to_uint8(array)

                tiff.imwrite(raw_path, arr16)
                tiff.imwrite(eight_path, arr8)

                _save_colorbar_figure(
                    array,
                    figure_directories["raw"] / f"{sanitized}_{label}.png",
                    title=f"{channel_name} - {label}",
                    subtitle="raw min/max",
                    color=channel_color,
                )
                _save_colorbar_figure(
                    array,
                    figure_directories["p95"] / f"{sanitized}_{label}.png",
                    title=f"{channel_name} - {label}",
                    subtitle="0–95th percentile",
                    color=channel_color,
                    scaling="percentile",
                    percentile=95.0,
                )
                _save_colorbar_figure(
                    array,
                    figure_directories["mad"] / f"{sanitized}_{label}.png",
                    title=f"{channel_name} - {label}",
                    subtitle="median±3·MAD",
                    color=channel_color,
                    scaling="mad",
                    mad_scale=3.0,
                )

    _write_figures_manifest(output_fig)


def _discover_channel_paths(handle: h5py.File) -> List[Tuple[int, h5py.Dataset]]:
    result: List[Tuple[int, h5py.Dataset]] = []
    dataset_group = handle.get("DataSet")
    if dataset_group is None:
        return result

    for key in dataset_group:
        if not key.startswith("ResolutionLevel"):
            continue
        res_group = dataset_group[key]
        time_group = res_group.get("TimePoint 0")
        if time_group is None:
            continue
        for channel_key in time_group:
            if not channel_key.startswith("Channel"):
                continue
            channel_group = time_group.get(channel_key)
            if channel_group is None or "Data" not in channel_group:
                continue
            try:
                index = int(channel_key.split()[-1])
            except ValueError:
                continue
            result.append((index, channel_group["Data"]))
        break  # only use the first resolution level
    return sorted(result, key=lambda item: item[0])


def _sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    cleaned = cleaned.strip("_.")
    return cleaned or "channel"


def _to_uint16(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize <= 2:
        return data.astype(np.uint16, copy=False)
    min_val = float(data.min())
    max_val = float(data.max())
    if max_val <= min_val:
        return np.zeros_like(data, dtype=np.uint16)
    normalized = (data - min_val) / (max_val - min_val)
    return np.clip(np.round(normalized * 65535.0), 0, 65535).astype(np.uint16)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    min_val = float(data.min())
    max_val = float(data.max())
    if max_val <= min_val:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - min_val) / (max_val - min_val)
    return np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)


def _save_colorbar_figure(
    array: np.ndarray,
    path: Path,
    *,
    title: str,
    subtitle: str,
    color: Tuple[float, float, float],
    scaling: str = "raw",
    percentile: float = 95.0,
    mad_scale: float = 3.0,
) -> None:
    data = np.asarray(array, dtype=np.float32)
    raw_min = float(data.min())
    raw_max = float(data.max())

    vmin, vmax = _determine_scale(
        data,
        mode=scaling,
        percentile=percentile,
        mad_scale=mad_scale,
        raw_min=raw_min,
        raw_max=raw_max,
    )
    color = tuple(max(0.0, min(1.0, float(c))) for c in color)
    cmap = LinearSegmentedColormap.from_list("channel_map", [(0.0, 0.0, 0.0), color])

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n{subtitle}: {vmin:.2f}–{vmax:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Pixel intensity")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _determine_scale(
    data: np.ndarray,
    *,
    mode: str,
    percentile: float,
    mad_scale: float,
    raw_min: float,
    raw_max: float,
) -> Tuple[float, float]:
    if raw_max <= raw_min:
        return raw_min, raw_max

    if mode == "percentile":
        top = float(np.percentile(data, percentile))
        if top <= raw_min:
            top = raw_max
        return raw_min, min(top, raw_max)

    if mode == "mad":
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        if mad == 0.0:
            return raw_min, raw_max
        vmin = max(raw_min, median - mad_scale * mad)
        vmax = min(raw_max, median + mad_scale * mad)
        if vmax <= vmin:
            return raw_min, raw_max
        return vmin, vmax

    return raw_min, raw_max


def _write_figures_manifest(root: Path) -> None:
    manifest_path = root / "README.txt"
    content = """Channel projection visualization guide
================================

This folder contains 3 subdirectories, each rendering the same channel
projections with different intensity scaling strategies. Underlying
TIFF exports in ../16bit and ../8bit remain untouched.

Subdirectories
--------------
raw_min_max/
    Displays the raw pixel range with no scaling beyond global min/max.
percentile_95/
    Intensity range is min → 95th percentile, emphasizing dim structures
    while compressing bright outliers. Pixel values remain raw underneath.
median_mad/
    Uses median ± 3 × MAD (median absolute deviation) for a robust view
    centred around typical intensities. Highlights very bright or very
    dim outliers less aggressively.

All PNGs are pseudo-coloured using the channel colour defined in the
Imaris metadata so that hue hints at the fluorophore while the scale
bar reflects original pixel units.
"""
    manifest_path.write_text(content.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
