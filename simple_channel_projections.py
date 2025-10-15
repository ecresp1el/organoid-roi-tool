"""
Batch export of raw max/mean/median projections for Imaris ``.ims`` files.

This script walks a directory, finds every ``.ims`` file, and writes three
per-channel projections (max, mean, median) without applying any additional
transformations. Results are saved alongside the input data under a
``simple_projections`` directory.
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

from imaris_tools.metadata import read_metadata

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
    channel_names = {channel.index: channel.name for channel in metadata.channels}

    output_root = path.parent / OUTPUT_FOLDER_NAME / path.stem
    output_16 = output_root / "16bit"
    output_8 = output_root / "8bit"
    output_fig = output_root / "figures"
    output_16.mkdir(parents=True, exist_ok=True)
    output_8.mkdir(parents=True, exist_ok=True)
    output_fig.mkdir(parents=True, exist_ok=True)

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

            channel_name = channel_names.get(channel_index, f"Channel_{channel_index:02d}")
            sanitized = _sanitize_name(channel_name)

            projections = [
                ("max", max_proj),
                ("mean", mean_proj),
                ("median", median_proj),
            ]

            for label, array in projections:
                raw_path = output_16 / f"{sanitized}_{label}.tif"
                eight_path = output_8 / f"{sanitized}_{label}.tif"
                figure_path = output_fig / f"{sanitized}_{label}.png"

                arr16 = _to_uint16(array)
                arr8 = _to_uint8(array)

                tiff.imwrite(raw_path, arr16)
                tiff.imwrite(eight_path, arr8)
                _save_colorbar_figure(array, figure_path, title=f"{channel_name} - {label}")


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


def _save_colorbar_figure(array: np.ndarray, path: Path, *, title: str) -> None:
    data = np.asarray(array, dtype=np.float32)
    vmin = float(data.min())
    vmax = float(data.max())

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    img = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\nmin={vmin:.2f}, max={vmax:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Pixel intensity")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
