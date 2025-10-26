#!/usr/bin/env python3
"""
Assemble a Cellpose training workspace by symlinking TIFFs referenced in the
CellProfiler/Cellpose metadata CSVs produced by
``prepare_for_cellprofiler_cellpose.py``.

The script reads either:

- ``cellpose_multichannel_metadata.csv`` (preferred) located under the
  ``cellpose_multichannel_zcyx`` tree, or
- ``cellprofilerandcellpose_metadata.csv`` for single-channel exports.

Filters such as analysis name, projection type, and experimental group ensure
only the desired subset is linked. Existing contents of the output folder can
optionally be cleared beforehand, keeping the repo workspace tidy even when the
underlying dataset evolves.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symlink TIFFs from CellProfiler/Cellpose metadata into a training directory."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to cellpose_multichannel_metadata.csv or cellprofilerandcellpose_metadata.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory that will contain per-group subfolders with symlinks.",
    )
    parser.add_argument(
        "--analysis",
        action="append",
        dest="analyses",
        default=None,
        help="Filter to specific analysis name(s). Repeat to include multiple analyses.",
    )
    parser.add_argument(
        "--projection",
        action="append",
        dest="projections",
        default=None,
        help="Filter to projection type(s), e.g. max, mean, median.",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        default=None,
        help="Filter to experimental group(s), e.g. WT, KO.",
    )
    parser.add_argument(
        "--group-subdirs",
        action="store_true",
        help="Place symlinks inside per-group subdirectories instead of a flat output folder.",
    )
    parser.add_argument(
        "--channel-slug",
        action="append",
        dest="channel_slugs",
        default=None,
        help="(Single-channel metadata only) include rows with these channel_slug values.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing contents of the output directory before linking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without creating any symlinks.",
    )
    return parser.parse_args()


def slugify(value: str, *, allow_period: bool = True) -> str:
    pattern = r"[^A-Za-z0-9._-]+" if allow_period else r"[^A-Za-z0-9_-]+"
    token = re.sub(pattern, "_", value.strip()).strip("_")
    return token or "item"


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)


def detect_metadata_kind(fieldnames: Iterable[str]) -> str:
    fields = set(fieldnames)
    if "channel_count" in fields:
        return "multi"
    return "single"


def ensure_columns(fieldnames: Iterable[str], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {', '.join(sorted(missing))}.")


def load_rows(
    csv_path: Path,
    analyses: Optional[Set[str]],
    projections: Optional[Set[str]],
    groups: Optional[Set[str]],
    channel_slugs: Optional[Set[str]],
) -> Tuple[str, List[Dict[str, str]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header detected in {csv_path}.")
        metadata_kind = detect_metadata_kind(reader.fieldnames)

        required = {"analysis", "group", "projection_type", "export_path", "sample_id"}
        ensure_columns(reader.fieldnames, required)
        if metadata_kind == "single":
            ensure_columns(reader.fieldnames, {"channel_slug"})

        rows: List[Dict[str, str]] = []
        for row in reader:
            analysis = (row.get("analysis") or "").strip()
            if analyses and analysis not in analyses:
                continue

            group = (row.get("group") or "").strip()
            if groups and group not in groups:
                continue

            projection = (row.get("projection_type") or "").strip()
            if projections and projection not in projections:
                continue

            if metadata_kind == "single" and channel_slugs:
                slug = (row.get("channel_slug") or "").strip()
                if slug not in channel_slugs:
                    continue

            rows.append(row)

    return metadata_kind, rows


def ensure_symlink(src: Path, dest: Path, *, dry_run: bool) -> bool:
    if not dest.parent.exists():
        if dry_run:
            print(f"[dry] mkdir -p {dest.parent}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        if dest.is_symlink():
            existing = dest.resolve()
            if existing == src:
                return False
            dest.unlink()
        else:
            raise FileExistsError(f"Cannot overwrite existing file: {dest}")

    if dry_run:
        print(f"[dry] ln -s {src} {dest}")
    else:
        os.symlink(src, dest)
    return True


def build_dest_name(row: Dict[str, str], src: Path) -> str:
    group = slugify(row.get("group", "group"), allow_period=False)
    sample = slugify(row.get("sample_id", "sample"))
    projection = slugify(row.get("projection_type", "proj"), allow_period=False)
    pieces = [group, sample, projection, src.stem]
    return "__".join(filter(None, pieces)) + src.suffix


def summarize(counts: Dict[str, int]) -> str:
    parts = [f"{group}={count}" for group, count in sorted(counts.items())]
    return ", ".join(parts) if parts else "none"


def main() -> int:
    args = parse_args()
    metadata_csv = args.metadata.expanduser().resolve()
    output_root = args.output.expanduser()

    if not metadata_csv.exists():
        print(f"[error] Metadata CSV not found: {metadata_csv}", file=sys.stderr)
        return 1

    analyses = set(args.analyses) if args.analyses else None
    projections = set(args.projections) if args.projections else None
    groups = set(args.groups) if args.groups else None
    channel_slugs = set(args.channel_slugs) if args.channel_slugs else None

    start = perf_counter()
    metadata_kind, rows = load_rows(metadata_csv, analyses, projections, groups, channel_slugs)

    if not rows:
        print("[warn] No metadata rows matched the provided filters.")
        return 0

    print(
        f"[info] Loaded {len(rows)} metadata rows ({metadata_kind}) "
        f"from {metadata_csv.name}"
    )

    if args.clear_output and not args.dry_run:
        clear_directory(output_root)

    linked = 0
    skipped = 0
    group_counts: Dict[str, int] = {}
    missing: List[str] = []

    for idx, row in enumerate(rows, start=1):
        src = Path((row.get("export_path") or "").strip()).expanduser()
        if not src.exists():
            missing.append(str(src))
            skipped += 1
            continue

        group = (row.get("group") or "unknown").strip() or "unknown"
        dest_dir = output_root / group if args.group_subdirs else output_root
        dest_name = build_dest_name(row, src)
        dest_path = dest_dir / dest_name

        try:
            created = ensure_symlink(src, dest_path, dry_run=args.dry_run)
        except FileExistsError as exc:
            print(f"[warn] {exc}")
            skipped += 1
            continue

        if created:
            linked += 1
            group_counts[group] = group_counts.get(group, 0) + 1
        else:
            skipped += 1

        if idx % 25 == 0:
            print(
                f"[prog] Processed {idx}/{len(rows)} rows | linked={linked} "
                f"| skipped={skipped}"
            )

    elapsed = perf_counter() - start
    print(
        f"[done] Linked {linked} file(s) in {elapsed:.2f}s "
        f"(skipped={skipped}; by-group: {summarize(group_counts)})"
    )

    if missing:
        print(f"[warn] {len(missing)} source file(s) missing. See log for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
