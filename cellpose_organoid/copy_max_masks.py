#!/usr/bin/env python3
"""Copy max-projection Cellpose masks into the mean/median folders.

The whole-organoid workflow draws masks on the ``max`` projection. This helper
reuses those masks for the ``mean`` and ``median`` projections so you can run
statistics without redrawing anything.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

TARGET_PROJECTIONS = ("mean", "median")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate *_seg.npy (and optional *_cp_masks.png) from the max projection "
            "into mean/median folders so multi-projection analyses can reuse the same ROI."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Project folder (top-level, same value passed to analyse_whole_organoid.py)",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        help="Analysis name, e.g. PCDHvsLHX6_WTvsKO_IHC or NestinvsDcx_WTvsKO_IHC",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=("WT", "KO"),
        help="Groups to process (default: WT KO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files, useful for validation.",
    )
    parser.add_argument(
        "--include-png",
        action="store_true",
        help="Also copy *_cp_masks.png files if present alongside *_seg.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    max_root = base_path / "cellprofilerandcellpose_folder" / "cellpose_multichannel_zcyx" / args.analysis / "max"
    if not max_root.exists():
        raise FileNotFoundError(f"Max projection folder missing: {max_root}")

    print(f"[INFO] Copying masks for analysis {args.analysis} under {base_path}")
    print(f"[INFO] Source (max) directory: {max_root}")

    seg_files = list(iter_seg_files(max_root, args.groups))
    if not seg_files:
        print("[WARN] No *_seg.npy files found under 'max'. Nothing to copy.")
        return

    suffixes = ["_seg.npy"]
    if args.include_png:
        suffixes.append("_cp_masks.png")

    copied = 0
    skipped = 0
    for src in seg_files:
        stem = src.stem
        for projection in TARGET_PROJECTIONS:
            dest_dir = src.parents[2] / projection / src.parent.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for suffix in suffixes:
                if suffix == "_seg.npy":
                    src_file = src
                    dest_name = src_file.name.replace("__max__", f"__{projection}__")
                    dest = dest_dir / dest_name
                    # Early versions of this helper produced names like
                    # ``*_seg_seg.npy``. Normalise those in-place before copy.
                    wrong_name = dest_dir / f"{stem}{suffix}"
                    if wrong_name.exists() and not dest.exists():
                        print(f"[FIX] Renaming legacy file {wrong_name.name} -> {dest.name}")
                        wrong_name.rename(dest)
                    legacy_name = dest_dir / src_file.name
                    if legacy_name.exists() and not dest.exists():
                        print(f"[FIX] Renaming legacy file {legacy_name.name} -> {dest.name}")
                        legacy_name.rename(dest)
                    if dest.exists():
                        print(f"[SKIP] {dest} already exists")
                        skipped += 1
                        continue
                    if args.dry_run:
                        print(f"[DRY] Copy {src_file} -> {dest}")
                    else:
                        shutil.copy2(src_file, dest)
                        print(f"[OK] Copied {dest}")
                        copied += 1
                else:
                    png_candidate = src.with_name(src.stem.replace("_seg", "") + suffix)
                    if not png_candidate.exists():
                        continue
                    src_file = png_candidate
                    dest_name = src_file.name.replace("__max__", f"__{projection}__")
                    dest = dest_dir / dest_name
                    legacy_name = dest_dir / src_file.name
                    if legacy_name.exists() and not dest.exists():
                        print(f"[FIX] Renaming legacy file {legacy_name.name} -> {dest.name}")
                        legacy_name.rename(dest)
                    if dest.exists():
                        print(f"[SKIP] {dest} already exists")
                        skipped += 1
                        continue
                    if args.dry_run:
                        print(f"[DRY] Copy {src_file} -> {dest}")
                    else:
                        shutil.copy2(src_file, dest)
                        print(f"[OK] Copied {dest}")
                        copied += 1

    print(f"[DONE] Masks processed. Files copied: {copied}, skipped: {skipped}")


def iter_seg_files(max_root: Path, groups: Iterable[str]) -> Iterable[Path]:
    """Yield each ``*_seg.npy`` file inside ``max/<group>/``.

    Printing missing group directories helps non-programmers understand why
    nothing was copied when a typo sneaks into ``--groups``.
    """

    for group in groups:
        group_dir = max_root / group
        if not group_dir.exists():
            print(f"[WARN] Missing group directory: {group_dir}")
            continue
        yield from sorted(group_dir.glob("*_seg.npy"))


if __name__ == "__main__":
    main()
