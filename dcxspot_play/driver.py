"""CLI and programmatic driver to batch the DCX Otsu pipeline across a project."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .otsu_pipeline import run_one


@dataclass
class RunSummary:
    inspected: int
    processed: int
    skipped: int
    output_root: Path


def _iter_mask_triplets(
    project_root: Path,
    fluor_subdir: str,
    fluor_suffix: str,
) -> Iterable[tuple[str, Path, Path, Path]]:
    wells_root = project_root / "wells"
    if not wells_root.is_dir():
        raise SystemExit(f"Project root {project_root} must contain a 'wells' directory")

    suffix = "_mask.tif"
    for mask_path in wells_root.rglob(f"*{suffix}"):
        base = mask_path.name[: -len(suffix)]
        bf_path = mask_path.parent / f"{base}.tif"
        mcherry_path = mask_path.parent / fluor_subdir / f"{base}{fluor_suffix}"
        yield base, mask_path, bf_path, mcherry_path


def run_project(
    project_root: Path,
    *,
    fluor_subdir: str = "fluorescence",
    fluor_suffix: str = "_mcherry.tif",
    output_root: Path | None = None,
    output_subdir: str | None = "dcxspot",
    limit: int | None = None,
    min_area: int = 24,
    max_area: int = 8000,
    min_distance: int = 3,
    morph_radius: int = 0,
    normalize_percentiles: tuple[float, float] = (0.0, 0.0),
    save_1x4: bool = False,
    verbose: bool = True,
) -> RunSummary:
    project_root = Path(project_root).expanduser().resolve()
    base_output_root = None
    if output_root is not None:
        base_output_root = Path(output_root).expanduser().resolve()
        base_output_root.mkdir(parents=True, exist_ok=True)

    inspected = 0
    processed = 0
    skipped_missing = 0

    for base, mask_path, bf_path, mcherry_path in _iter_mask_triplets(
        project_root,
        fluor_subdir,
        fluor_suffix,
    ):
        if limit is not None and inspected >= limit:
            break
        inspected += 1

        fluor_folder = mask_path.parent / fluor_subdir
        if not fluor_folder.exists():
            if verbose:
                print(f"[skip] missing fluorescence folder: {fluor_folder}")
            skipped_missing += 1
            continue

        if not mcherry_path.exists():
            if verbose:
                print(f"[skip] missing fluorescence file: {mcherry_path}")
            skipped_missing += 1
            continue
        if not bf_path.exists():
            if verbose:
                print(f"[skip] missing BF image: {bf_path}")
            skipped_missing += 1
            continue

        if output_subdir is not None:
            out_dir = mask_path.parent / output_subdir
        elif base_output_root is not None:
            rel_dir = mask_path.parent.relative_to(project_root)
            out_dir = base_output_root / rel_dir
        else:
            rel_dir = mask_path.parent.relative_to(project_root)
            fallback_root = project_root / "dcxspot_outputs"
            out_dir = fallback_root / rel_dir
            fallback_root.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "output_dir": out_dir,
            "image_id": base,
            "min_area": min_area,
            "max_area": max_area,
            "min_distance": min_distance,
            "morph_radius": morph_radius,
            "normalize_percentiles": normalize_percentiles,
            "save_1x4": save_1x4,
        }

        if verbose:
            print(f"[run] {base} -> {out_dir}")
        result = run_one(
            mcherry_path=mcherry_path,
            roi_path=mask_path,
            bf_path=bf_path,
            params=params,
        )
        processed += 1
        if verbose:
            print(
                f"      clusters={result['qc']['n_final']}  panel={result['panel_1x3_path']}"
            )

    if output_subdir is not None:
        final_output = project_root
    elif base_output_root is not None:
        final_output = base_output_root
    else:
        final_output = project_root / "dcxspot_outputs"

    if verbose:
        print("\nSummary")
        print(f"  inspected : {inspected}")
        print(f"  processed : {processed}")
        print(f"  skipped   : {skipped_missing}")
        print(f"  outputs   : {final_output}")

    return RunSummary(
        inspected=inspected,
        processed=processed,
        skipped=skipped_missing,
        output_root=final_output,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_root", type=Path, help="ROI project root containing wells/")
    ap.add_argument(
        "--fluor-subdir",
        default="fluorescence",
        help="Subfolder (inside each timepoint) holding fluorescence TIFFs",
    )
    ap.add_argument(
        "--fluor-suffix",
        default="_mcherry.tif",
        help="Suffix appended to base name to locate the fluorescence image",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root folder to store DCX outputs (defaults to <project_root>/dcxspot_outputs)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N masks")
    ap.add_argument("--min-area", type=int, default=24)
    ap.add_argument("--max-area", type=int, default=8000)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--morph-radius", type=int, default=0)
    ap.add_argument(
        "--normalize-percentiles",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(0.0, 0.0),
        help="Percentiles for optional stretch prior to Otsu (0 0 disables)",
    )
    ap.add_argument("--save-1x4", action="store_true", help="Also save the 1x4 QC panel")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_project(
        project_root=args.project_root,
        fluor_subdir=args.fluor_subdir,
        fluor_suffix=args.fluor_suffix,
        output_root=args.output_root,
        limit=args.limit,
        min_area=args.min_area,
        max_area=args.max_area,
        min_distance=args.min_distance,
        morph_radius=args.morph_radius,
        normalize_percentiles=tuple(args.normalize_percentiles),
        save_1x4=args.save_1x4,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
