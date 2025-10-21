"""Export 16-bit projection TIFFs for CellProfiler/Cellpose ingestion.

This script represents the third stage of the imaging pipeline:

1. ``simple_channel_projections.py`` generates max/mean/median projections.
2. ``run_projection_analysis.py`` runs a domain-specific analysis (e.g.
   PCDHvsLHX6 or NestinvsDcx), recording metadata about every projection.
3. This script gathers the 16-bit TIFF paths from the analysis manifests and
   organises them into ``cellprofilerandcellpose_folder`` grouped by analysis,
   channel, and experimental group (WT vs KO). A metadata CSV accompanies the
   exports so downstream tools can trace each file back to its origin.

If the projection exports or analysis results are missing, the script prints
clear guidance and aborts so scientists can complete the earlier steps first.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_ANALYSES = [
    "PCDHvsLHX6_WTvsKO_IHC",
    "NestinvsDcx_WTvsKO_IHC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 16-bit projection TIFFs for automated segmentation tools. "
            "Requires that simple_channel_projections and run_projection_analysis "
            "have already been executed for the target dataset."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Project directory that contains simple_projections and analysis_results.",
    )
    parser.add_argument(
        "--analysis",
        dest="analyses",
        action="append",
        default=None,
        help=(
            "Name of the analysis to export (defaults to all known analyses). "
            "Repeat to export multiple analyses."
        ),
    )
    return parser.parse_args()


def ensure_step_one(base_path: Path) -> Path:
    projections = base_path / "simple_projections"
    if not projections.exists() or not any(projections.iterdir()):
        raise FileNotFoundError(
            f"simple_projections not found at {projections}. Run simple_channel_projections.py first."
        )
    print(f"[check] Found projection exports in {projections}")
    return projections


def ensure_step_two(base_path: Path, analysis: str) -> Path:
    analysis_root = base_path / "analysis_results" / analysis
    pipeline_root = analysis_root / "analysis_pipeline"
    if not pipeline_root.exists() or not any(pipeline_root.iterdir()):
        raise FileNotFoundError(
            f"No analysis pipeline outputs at {pipeline_root}. "
            "Run run_projection_analysis.py for this analysis before exporting."
        )
    print(f"[check] Found analysis outputs in {pipeline_root}")
    return pipeline_root


def iter_manifests(pipeline_root: Path) -> Iterable[tuple[str, Path, pd.DataFrame]]:
    for channel_dir in sorted(pipeline_root.iterdir()):
        if not channel_dir.is_dir():
            continue
        data_dir = channel_dir / "data"
        manifest_path = data_dir / "manifest.csv"
        if not manifest_path.exists():
            continue
        try:
            df = pd.read_csv(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to read {manifest_path}: {exc}")
            continue
        yield channel_dir.name, manifest_path, df


def prepare_exports(
    base_path: Path,
    analysis: str,
    projections_root: Path,
    pipeline_root: Path,
    output_root: Path,
) -> List[dict]:
    records: List[dict] = []
    for channel_slug, manifest_path, manifest in iter_manifests(pipeline_root):
        channel_output = output_root / analysis / channel_slug
        channel_output.mkdir(parents=True, exist_ok=True)

        print(
            f"[export] Processing manifest {manifest_path} "
            f"({len(manifest)} rows) -> {channel_output}"
        )

        required_cols = {
            "group",
            "sample_id",
            "channel",
            "channel_canonical",
            "channel_marker",
            "path",
            "projection_type",
        }
        missing = required_cols - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {sorted(missing)}."
            )

        for row in manifest.itertuples():
            source_path = Path(row.path)
            if not source_path.exists():
                print(f"[warn] Skipping missing TIFF: {source_path}")
                continue

            group = getattr(row, "group", "unknown")
            group_dir = channel_output / str(group)
            group_dir.mkdir(parents=True, exist_ok=True)

            dest_name = (
                f"{row.sample_id}__{row.channel_marker}_{row.projection_type}.tif"
            )
            destination = group_dir / dest_name
            shutil.copy2(source_path, destination)

            records.append(
                {
                    "analysis": analysis,
                    "channel_slug": channel_slug,
                    "channel": row.channel,
                    "channel_canonical": getattr(row, "channel_canonical", row.channel),
                    "channel_marker": getattr(row, "channel_marker", row.channel),
                    "channel_wavelength_nm": getattr(row, "channel_wavelength_nm", ""),
                    "group": group,
                    "sample_id": row.sample_id,
                    "projection_type": row.projection_type,
                    "filename": getattr(row, "filename", source_path.name),
                    "source_path": str(source_path),
                    "export_path": str(destination),
                    "manifest": str(manifest_path),
                }
            )

    return records


def write_metadata(output_root: Path, records: List[dict]) -> None:
    if not records:
        print("[export] No files exported; skipping metadata CSV.")
        return

    csv_path = output_root / "cellprofilerandcellpose_metadata.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[done] Metadata written to {csv_path}")


def main() -> int:
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    analyses = args.analyses or DEFAULT_ANALYSES

    print(f"[info] Preparing exports under {base_path}")
    projections_root = ensure_step_one(base_path)

    output_root = base_path / "cellprofilerandcellpose_folder"
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []

    for analysis in analyses:
        print("=" * 80)
        print(f"[analysis] {analysis}")
        try:
            pipeline_root = ensure_step_two(base_path, analysis)
        except FileNotFoundError as exc:
            print(f"[warn] Skipping {analysis}: {exc}")
            continue

        records = prepare_exports(
            base_path=base_path,
            analysis=analysis,
            projections_root=projections_root,
            pipeline_root=pipeline_root,
            output_root=output_root,
        )
        all_records.extend(records)
        print(f"[analysis] {analysis} -> exported {len(records)} TIFF(s).")

    write_metadata(output_root, all_records)
    print(f"[done] Export folder ready at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
