#!/usr/bin/env python3
"""
Draw whole-organoid masks by hand and save Cellpose-compatible outputs.

Example:
    conda activate organoid_roi_incucyte_imaging
    python cellpose_organoid/manual_mask_gui.py \\
        --input-dir /Volumes/.../cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max

For each TIFF the script opens a napari viewer. Add one or more polygon shapes
that outline the entire organoid, then press ``S`` (or close the window) to save.
The script writes ``*_seg.npy`` and ``*_cp_masks.png`` next to the TIFF with the
same field layout Cellpose generates, so downstream tooling treats the files
exactly like automatic outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import napari  # type: ignore
import numpy as np
import tifffile
from skimage.draw import polygon
from skimage.measure import label

from qtpy import QtWidgets

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual Cellpose-style whole-organoid mask GUI. "
            "If no directory is supplied, a file dialog will prompt you to pick one."
        )
    )
    parser.add_argument(
        "--dir",
        "--input-dir",
        dest="input_dir",
        type=Path,
        default=None,
        help="Directory containing TIFF images (e.g. cellpose_multichannel_zcyx/.../max). "
        "Optional when using the GUI picker.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Process a single TIFF image (overrides --dir).",
    )
    parser.add_argument(
        "--glob",
        default="*.tif",
        help="Glob pattern for images (default: *.tif, case-insensitive variants handled automatically).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index to display when the TIFF has multiple channels. "
        "If omitted, the script displays a max projection.",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=1500.0,
        help="Diameter metadata stored in the *_seg.npy payload (matches Cellpose defaults).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_seg.npy files (default: skip already-labelled images).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --input-dir for matching TIFFs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = resolve_image_paths(args)
    if not image_paths:
        raise FileNotFoundError("No TIFF images found. Provide --dir/--image-path or pick a folder in the dialog.")

    print(
        "[INFO] Manual mask GUI controls:\n"
        "  • Use the Shapes layer (Add polygon) to outline the organoid.\n"
        "  • Press 'S' or close the window to save and move to the next image.\n"
        "  • Leave the canvas empty and close the window to skip.\n"
    )

    for idx, img_path in enumerate(image_paths, start=1):
        seg_path = img_path.with_name(img_path.stem + "_seg.npy")
        mask_png_path = img_path.with_name(img_path.stem + "_cp_masks.png")

        if seg_path.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}/{len(image_paths)}) {img_path.name} already labelled.")
            continue

        print(f"[OPEN] ({idx}/{len(image_paths)}) {img_path}")
        data = tifffile.imread(img_path)
        display = prepare_display_image(data, args.channel)

        viewer = napari.Viewer()
        viewer.title = f"Manual mask: {img_path.name}"
        viewer.add_image(display, name="image", contrast_limits=auto_contrast_limits(display))
        shapes = viewer.add_shapes(name="mask", shape_type="polygon", edge_color="yellow", face_color="transparent")
        viewer.bind_key("s", lambda _: viewer.close())
        napari.run()

        polygons = list(shapes.data)
        viewer.close()

        if not polygons:
            print(f"[WARN] No mask drawn for {img_path.name}; skipping.")
            continue

        mask_bool = polygons_to_mask(polygons, display.shape)
        mask_labels = label(mask_bool).astype(np.uint16)
        if mask_labels.max() == 0:
            print(f"[WARN] Mask for {img_path.name} contains no pixels; skipping.")
            continue

        save_cellpose_payload(
            seg_path,
            mask_labels,
            img_path=img_path,
            diameter=args.diameter,
            channel_index=args.channel,
        )
        imageio.imwrite(mask_png_path, mask_labels.astype(np.uint16))
        print(f"[OK] Saved {seg_path.name} and {mask_png_path.name}")

    print("[DONE] Manual mask session complete.")


def resolve_image_paths(args: argparse.Namespace) -> List[Path]:
    if args.image_path is not None:
        path = args.image_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return [path]

    root = args.input_dir.expanduser().resolve() if args.input_dir else prompt_for_directory()
    if root is None:
        return []

    if root.is_file():
        return [root]

    return list(iter_image_paths(root, args.glob, args.recursive))


def iter_image_paths(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []

    patterns = {pattern}
    if pattern.lower().endswith(".tif"):
        patterns.add(pattern + "f")

    globber = root.rglob if recursive else root.glob
    seen: set[Path] = set()
    for pat in patterns:
        for path in sorted(globber(pat)):
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and path not in seen:
                seen.add(path)
                yield path


def prompt_for_directory() -> Optional[Path]:
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication([])
        created_app = True

    selected = QtWidgets.QFileDialog.getExistingDirectory(
        None,
        "Select folder containing TIFF images",
        str(Path.home()),
    )

    if created_app:
        # We created a temporary QApp solely for the dialog; no need to keep it alive.
        app.quit()

    if selected:
        return Path(selected).expanduser().resolve()
    return None


def prepare_display_image(array: np.ndarray, channel_index: Optional[int]) -> np.ndarray:
    arr = np.squeeze(array)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # try channel-first
        if channel_index is not None:
            if channel_index < arr.shape[0]:
                return arr[channel_index]
            if channel_index < arr.shape[-1]:
                return arr[..., channel_index]
        # fallback to max projection
        return arr.max(axis=0) if arr.shape[0] <= arr.shape[-1] else arr.max(axis=-1)
    raise ValueError(f"Unsupported image dimensions {arr.shape}")


def auto_contrast_limits(image: np.ndarray) -> Sequence[float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return (0.0, 1.0)
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(finite.min())
        high = float(finite.max() if finite.max() > low else low + 1.0)
    return (low, high)


def polygons_to_mask(polygons: Sequence[np.ndarray], shape: Sequence[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for poly in polygons:
        if poly.ndim != 2 or poly.shape[1] != 2:
            continue
        rr, cc = polygon(poly[:, 0], poly[:, 1], shape)
        mask[rr, cc] = True
    return mask


def save_cellpose_payload(
    seg_path: Path,
    mask_labels: np.ndarray,
    *,
    img_path: Path,
    diameter: float,
    channel_index: Optional[int],
) -> None:
    payload = {
        "outlines": [],
        "colors": [],
        "masks": mask_labels,
        "chan_choose": [channel_index if channel_index is not None else 0, 0],
        "filename": str(img_path),
        "flows": [],
        "ismanual": True,
        "manual_changes": True,
        "model_path": "manual_mask_gui",
        "flow_threshold": None,
        "cellprob_threshold": None,
        "normalize_params": None,
        "restore": False,
        "ratio": 1.0,
        "diameter": float(diameter),
    }
    np.save(seg_path, payload, allow_pickle=True)


if __name__ == "__main__":
    main()
