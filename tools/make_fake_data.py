#!/usr/bin/env python
"""
Generate synthetic TIFF images for testing the Organoid ROI Tool.

By default, creates 2D grayscale images with a bright circular blob
on a dim background, saved as 16-bit TIFF with resolution tags so the
GUI can populate pixel_size_um.

Examples (run inside the Conda env):

  # Create a flat "raw" folder with Incucyte-like names
  python tools/make_fake_data.py --raw sample_raw --wells A01 A02 A03 \
    --days 01 --times 00:00 12:00

  # Then reorganize into a project
  python reorganize.py --raw sample_raw --out sample_project

  # Or directly create organized structure
  python tools/make_fake_data.py --organized sample_project \
    --wells A01 A02 --days 01 02 --times 00:00
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import tifffile as tiff


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _make_blob_image(
    h: int = 512,
    w: int = 512,
    radius: float = 60.0,
    center: tuple[float, float] | None = None,
    noise_sigma: float = 20.0,
    dtype=np.uint16,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    if center is None:
        cy = h / 2 + np.random.uniform(-h * 0.1, h * 0.1)
        cx = w / 2 + np.random.uniform(-w * 0.1, w * 0.1)
    else:
        cy, cx = center
    rr = np.hypot(yy - cy, xx - cx)

    # Smooth “organism” profile: bright center, soft edges
    shell = np.clip(1.0 - np.tanh((rr - radius) / 5.0), 0.0, 1.0)
    base = 1000.0  # background
    img = base + 4000.0 * shell
    img += np.random.normal(0.0, noise_sigma, img.shape)

    img = np.clip(img, 0, np.iinfo(dtype).max).astype(dtype)
    return img


def _save_tiff_with_resolution(path: Path, data: np.ndarray, um_per_px: float = 10.0) -> None:
    # Convert micrometers per pixel to PPI (pixels per inch) using: um_per_px = 25400 / ppi
    # So ppi = 25400 / um_per_px
    ppi = 25400.0 / float(um_per_px)
    # Use same X/Y resolution; unit = inch
    tiff.imwrite(
        str(path),
        data,
        photometric="minisblack",
        resolution=(ppi, ppi),
        resolutionunit="INCH",
    )


def _to_time_str(hhmm: str) -> str:
    # Input like "12:00" -> "12h00m"
    hh, mm = hhmm.split(":")
    return f"{int(hh):02d}h{int(mm):02d}m"


def make_raw_names(raw_dir: Path, wells: list[str], days: list[str], times: list[str]) -> list[Path]:
    out: list[Path] = []
    for well in wells:
        for day in days:
            for t in times:
                name = f"synthetic_{well}_1_{day}d{_to_time_str(t)}.tif"
                out.append(raw_dir / name)
    return out


def make_organized_paths(root: Path, wells: list[str], days: list[str], times: list[str]) -> list[Path]:
    out: list[Path] = []
    for well in wells:
        for day in days:
            for t in times:
                day_dir = root / "wells" / well / f"day_{day}" / _to_time_str(t)
                out.append(day_dir / f"synthetic_{well}_{day}_{t.replace(':','')}.tif")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate synthetic TIFF data for Organoid ROI Tool")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw", type=str, help="Create flat folder with Incucyte-like names")
    group.add_argument("--organized", type=str, help="Create organized wells/<well>/day_XX/HHhMMm")
    ap.add_argument("--wells", nargs="+", default=["A01", "A02", "A03"], help="Well IDs (e.g., A01 A02)")
    ap.add_argument("--days", nargs="+", default=["01"], help="Day strings like 01 02 03")
    ap.add_argument("--times", nargs="+", default=["00:00", "12:00"], help="Time strings like HH:MM")
    ap.add_argument("--size", type=int, default=512, help="Image width/height in pixels")
    ap.add_argument("--radius", type=float, default=60.0, help="Approximate blob radius in pixels")
    ap.add_argument("--um_per_px", type=float, default=10.0, help="Micrometers per pixel written to TIFF tags")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.raw:
        raw_dir = Path(args.raw).expanduser().resolve()
        _ensure_dir(raw_dir)
        files = make_raw_names(raw_dir, args.wells, args.days, args.times)
    else:
        root = Path(args.organized).expanduser().resolve()
        files = make_organized_paths(root, args.wells, args.days, args.times)
        for p in files:
            _ensure_dir(p.parent)

    print(f"[make_fake_data] Writing {len(files)} TIFF images...")
    for p in files:
        img = _make_blob_image(h=args.size, w=args.size, radius=args.radius)
        _save_tiff_with_resolution(p, img, um_per_px=args.um_per_px)
        print(f"  + {p}")
    print("[make_fake_data] Done.")


if __name__ == "__main__":
    main()

