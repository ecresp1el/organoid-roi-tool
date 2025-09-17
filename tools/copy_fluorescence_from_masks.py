#!/usr/bin/env python3
"""Copy fluorescence TIFFs into an organized ROI project and apply existing masks.

The script searches a project folder for ROI mask files (``*_mask.tif``) that were
created by the organoid ROI tool. For every mask, it looks up the matching
fluorescence image in a source directory (flat folder of TIFFs) and then:

1. Copies the raw fluorescence TIFF into the same project timepoint folder,
   inside a channel-specific subdirectory.
2. Applies the ROI mask to generate a full-size masked fluorescence TIFF.
3. Crops the masked fluorescence image to the ROI bounding box and saves it.

No files are deleted or moved from their original locations; fluorescence TIFFs
are copied into the project. Re-running the script is safe: it can skip existing
outputs unless ``--overwrite`` is provided.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - validated at runtime
    raise SystemExit(
        "tifffile is required. Activate the organoid ROI environment or install it."
    ) from exc

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


@dataclass
class JobOutcome:
    mask_path: Path
    fluorescence_source: Optional[Path]
    skipped_reason: Optional[str] = None
    copied_raw: Optional[Path] = None
    saved_masked: Optional[Path] = None
    saved_cropped: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None


SUFFIX_MASK = "_mask.tif"
TIFF_EXTS = (".tif", ".tiff", ".TIF", ".TIFF")


def iter_masks(project_root: Path) -> Iterable[Path]:
    """Yield every mask path under ``wells`` with the expected suffix."""
    masks = project_root / "wells"
    if not masks.is_dir():
        raise SystemExit(f"Project path {project_root} does not contain a 'wells' folder")
    for path in masks.rglob(f"*{SUFFIX_MASK}"):
        if path.name.endswith("_roi_masked.tif") or path.name.endswith("_roi_masked_cropped.tif"):
            continue
        yield path


def read_tiff_array(path: Path) -> np.ndarray:
    """Read a TIFF into a numpy array with optional Pillow fallback."""
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        needs_codecs = "imagecodecs" in str(exc)
        if needs_codecs and Image is not None:
            with Image.open(path) as img:
                return np.array(img)
        raise


def load_mask(mask_path: Path) -> np.ndarray:
    mask = read_tiff_array(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Mask at {mask_path} is not 2-D (shape={mask.shape})")
    return mask > 0


def apply_mask(data: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return np.where(mask_bool, data, 0)
    expand_shape = (1,) * (data.ndim - 2) + mask_bool.shape
    mask_expanded = mask_bool.reshape(expand_shape)
    return np.where(mask_expanded, data, 0)


def crop_to_mask(data: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask is empty; nothing to crop")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return data[..., y0:y1, x0:x1]


def resolve_source(base_name: str, fluor_root: Path) -> Optional[Path]:
    for ext in TIFF_EXTS:
        candidate = fluor_root / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_mask(
    mask_path: Path,
    fluor_root: Path,
    target_subdir: str,
    channel: str,
    overwrite: bool,
    dry_run: bool,
) -> JobOutcome:
    base = mask_path.name[: -len(SUFFIX_MASK)]
    source = resolve_source(base, fluor_root)
    if not source:
        return JobOutcome(mask_path=mask_path, fluorescence_source=None, skipped_reason="source not found")

    timepoint_dir = mask_path.parent
    dest_dir = timepoint_dir / target_subdir
    raw_dest = dest_dir / f"{base}_{channel}.tif"
    masked_dest = dest_dir / f"{base}_{channel}_masked.tif"
    cropped_dest = dest_dir / f"{base}_{channel}_masked_cropped.tif"

    if not overwrite and all(p.exists() for p in (raw_dest, masked_dest, cropped_dest)):
        return JobOutcome(
            mask_path=mask_path,
            fluorescence_source=source,
            skipped_reason="outputs already exist",
        )

    outcome = JobOutcome(mask_path=mask_path, fluorescence_source=source)

    if dry_run:
        return outcome

    mask_bool = load_mask(mask_path)
    data = read_tiff_array(source)

    dest_dir.mkdir(parents=True, exist_ok=True)

    if overwrite or not raw_dest.exists():
        shutil.copy2(source, raw_dest)
        outcome.copied_raw = raw_dest

    if data.shape[-2:] != mask_bool.shape:
        raise ValueError(
            f"Fluorescence image shape {data.shape[-2:]} does not match mask {mask_bool.shape}"
        )

    masked = apply_mask(data, mask_bool)
    if overwrite or not masked_dest.exists():
        tifffile.imwrite(str(masked_dest), masked)
        outcome.saved_masked = masked_dest

    cropped = crop_to_mask(masked, mask_bool)
    if overwrite or not cropped_dest.exists():
        tifffile.imwrite(str(cropped_dest), cropped)
        outcome.saved_cropped = cropped_dest

    return outcome


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project", type=Path, help="Path to the ROI project folder (contains wells/)")
    ap.add_argument("fluorescence", type=Path, help="Path to the folder containing fluorescence TIFFs")
    ap.add_argument(
        "--channel",
        default="mcherry",
        help="Channel label to embed in output filenames (default: mcherry)",
    )
    ap.add_argument(
        "--target-subdir",
        default="fluorescence",
        help="Name of subdirectory (inside each timepoint folder) for fluorescence outputs",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fluorescence outputs if they already exist",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N masks (useful for testing)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying or writing files",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    project = args.project.expanduser().resolve()
    fluor_root = args.fluorescence.expanduser().resolve()

    if not project.exists():
        raise SystemExit(f"Project path does not exist: {project}")
    if not fluor_root.exists():
        raise SystemExit(f"Fluorescence path does not exist: {fluor_root}")

    outcomes: list[JobOutcome] = []
    for idx, mask_path in enumerate(iter_masks(project)):
        if args.limit is not None and idx >= args.limit:
            break
        outcome = process_mask(
            mask_path=mask_path,
            fluor_root=fluor_root,
            target_subdir=args.target_subdir,
            channel=args.channel,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        outcomes.append(outcome)
        status = "OK" if outcome.ok else f"SKIP ({outcome.skipped_reason})"
        print(f"[{status}] {mask_path.relative_to(project)}")

    total = len(outcomes)
    done = sum(1 for o in outcomes if o.ok)
    skipped = total - done
    print("\nSummary:")
    print(f"  Masks processed : {done}")
    print(f"  Masks skipped   : {skipped}")
    print(f"  Total inspected : {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
