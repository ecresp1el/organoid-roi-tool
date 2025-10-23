"""Export 16-bit projection TIFFs for CellProfiler/Cellpose ingestion.

This script represents the third stage of the imaging pipeline:

1. ``simple_channel_projections.py`` generates max/mean/median projections.
2. ``run_projection_analysis.py`` runs a domain-specific analysis (e.g.
   PCDHvsLHX6 or NestinvsDcx), recording metadata about every projection.
3. This script gathers the 16-bit TIFF paths from the analysis manifests and
   organises them into ``cellprofilerandcellpose_folder`` grouped by analysis,
   channel, and experimental group (WT vs KO). A metadata CSV accompanies the
   exports so downstream tools can trace each file back to its origin. The
   script now also emits multi-channel ZCYX stacks (one TIFF per sample) under
   ``cellpose_multichannel_zcyx`` so Cellpose can ingest combined channels
   directly. When available, the DAPI reference channel is copied alongside the
   analysis markers so the segmentation tools always receive a nuclear channel.

If the projection exports or analysis results are missing, the script prints
clear guidance and aborts so scientists can complete the earlier steps first.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "pandas is required for prepare_for_cellprofiler_cellpose.py. "
        "Install it in the active environment (e.g. `conda install pandas`)"
    ) from exc

try:  # pragma: no cover - runtime dependency check
    import tifffile
except ImportError:  # pragma: no cover - handled gracefully at runtime
    tifffile = None

DEFAULT_ANALYSES = [
    "PCDHvsLHX6_WTvsKO_IHC",
    "NestinvsDcx_WTvsKO_IHC",
]


_DAPI_TOKENS = {
    "dapi",
    "confocal_blue",
    "confocalblue",
    "blue",
    "nuclei",
}


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

    dapi_records = extend_with_dapi_exports(
        records,
        projections_root=projections_root,
        analysis=analysis,
        output_root=output_root,
    )
    if dapi_records:
        records.extend(dapi_records)
        print(
            f"[export] Added {len(dapi_records)} supplemental DAPI projection(s) for {analysis}."
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


def _slugify_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return token or "value"


def _ensure_zyx(arr: np.ndarray, path: Path) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim == 3:
        min_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, min_axis, 0)
        return arr

    raise ValueError(
        f"Unsupported TIFF dimensions {arr.shape} in {path}. Expected 2D or 3D arrays."
    )


def _record_is_dapi(record: dict) -> bool:
    for key in ("channel_marker", "channel_canonical", "channel", "channel_slug"):
        value = record.get(key)
        if isinstance(value, str) and _text_matches_dapi(value):
            return True
    return False


def _discover_dapi_projections(sample_dir: Path) -> Dict[str, Path]:
    matches: Dict[str, Path] = {}
    for tif_path in sample_dir.glob("*.tif"):
        stem = tif_path.stem
        projection_match = re.search(r"_(max|mean|median)(?=\.|$)", stem.lower())
        if not projection_match:
            continue
        projection_type = projection_match.group(1)
        channel_name = stem[: -(len(projection_type) + 1)]
        if not _text_matches_dapi(channel_name):
            continue
        matches.setdefault(projection_type, tif_path)
    return matches


def _text_matches_dapi(value: str) -> bool:
    slug = _slugify_token(value).lower()
    plain = re.sub(r"[^0-9a-z]+", "", value.lower())
    return slug in _DAPI_TOKENS or plain in _DAPI_TOKENS


def extend_with_dapi_exports(
    existing_records: List[dict],
    *,
    projections_root: Path,
    analysis: str,
    output_root: Path,
) -> List[dict]:
    if not projections_root.exists():
        return []

    supplemental: List[dict] = []

    existing_keys = {
        (record["sample_id"], record["projection_type"])
        for record in existing_records
        if _record_is_dapi(record)
    }

    sample_map: Dict[str, Dict[str, object]] = {}
    for record in existing_records:
        sample_id = record["sample_id"]
        info = sample_map.setdefault(sample_id, {"group": record["group"], "projections": set()})
        if record["group"] != info["group"]:  # pragma: no cover - data quality guard
            print(
                f"[warn] Sample {sample_id} has conflicting group assignments: {record['group']} vs {info['group']}.",
            )
        projections_set = info["projections"]
        assert isinstance(projections_set, set)
        projections_set.add(record["projection_type"])

    for sample_id, info in sorted(sample_map.items()):
        projections_set = info["projections"]
        assert isinstance(projections_set, set)
        sample_dir = projections_root / sample_id / "16bit"
        if not sample_dir.exists():
            continue

        dapi_by_projection = _discover_dapi_projections(sample_dir)
        if not dapi_by_projection:
            continue

        group_value = info["group"]
        for projection_type in sorted(projections_set):
            key = (sample_id, projection_type)
            if key in existing_keys:
                continue

            dapi_path = dapi_by_projection.get(projection_type)
            if dapi_path is None:
                print(
                    f"[warn] Missing DAPI {projection_type} projection for sample {sample_id} in {sample_dir}.",
                )
                continue

            channel_slug = "DAPI_reference"
            dest_dir = output_root / analysis / channel_slug / str(group_value)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"{sample_id}__DAPI_{projection_type}.tif"
            destination = dest_dir / dest_name

            shutil.copy2(dapi_path, destination)

            supplemental.append(
                {
                    "analysis": analysis,
                    "channel_slug": channel_slug,
                    "channel": "DAPI",
                    "channel_canonical": "DAPI",
                    "channel_marker": "DAPI",
                    "channel_wavelength_nm": "",
                    "group": group_value,
                    "sample_id": sample_id,
                    "projection_type": projection_type,
                    "filename": dapi_path.name,
                    "source_path": str(dapi_path),
                    "export_path": str(destination),
                    "manifest": "",
                }
            )

    return supplemental


def create_multichannel_stacks(output_root: Path, records: List[dict]) -> None:
    if not records:
        print("[cellpose] No exports available for multi-channel merge; skipping.")
        return

    if tifffile is None:
        print(
            "[warn] Skipping multi-channel Cellpose exports: tifffile is not available in this environment."
        )
        return

    dest_root = output_root / "cellpose_multichannel_zcyx"
    dest_root.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for record in records:
        key = (
            record["analysis"],
            record["group"],
            record["sample_id"],
            record["projection_type"],
        )
        grouped.setdefault(key, []).append(record)

    merged_rows: List[dict] = []

    for group_key, items in sorted(grouped.items()):
        analysis, group, sample_id, projection_type = group_key
        channel_names = _channel_summary(items)

        if len(items) < 2:
            print(
                f"[cellpose] Skipping {analysis}/{sample_id}/{projection_type}: "
                f"requires at least two channels (found: {channel_names or 'none'})."
            )
            continue

        if not any(_record_is_dapi(record) for record in items):
            print(
                f"[warn] Skipping {analysis}/{sample_id}/{projection_type}: "
                f"no DAPI channel detected among {channel_names or 'unknown channels'}."
            )
            continue

        channel_arrays: List[np.ndarray] = []
        channel_records: List[dict] = []

        for record in sorted(
            items,
            key=lambda rec: (
                rec.get("channel_canonical", ""),
                rec.get("channel_marker", ""),
                rec.get("channel_slug", ""),
            ),
        ):
            export_path = Path(record["export_path"])
            try:
                arr = np.asarray(tifffile.imread(export_path))
                zyx = _ensure_zyx(arr, export_path)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[warn] Failed to load {export_path}: {exc}")
                channel_arrays = []
                break

            channel_arrays.append(zyx)
            channel_records.append(record)

        if not channel_arrays:
            continue

        z_sizes = {arr.shape[0] for arr in channel_arrays}
        spatial_shapes = {arr.shape[1:] for arr in channel_arrays}
        if len(z_sizes) != 1 or len(spatial_shapes) != 1:
            print(
                f"[warn] Mismatched shapes for {analysis}/{sample_id}/{projection_type}; "
                f"Z planes: {sorted(z_sizes)}, spatial: {sorted(spatial_shapes)}."
            )
            continue

        stack = np.stack(channel_arrays, axis=1)  # Z x C x Y x X

        dest_dir = dest_root / analysis / projection_type / str(group)
        dest_dir.mkdir(parents=True, exist_ok=True)

        channel_labels = [
            _slugify_token(
                rec.get("channel_marker")
                or rec.get("channel_canonical")
                or rec.get("channel")
            )
            for rec in channel_records
        ]
        channel_descriptor = "+".join(channel_labels)
        dest_name = (
            f"{_slugify_token(str(sample_id))}__"
            f"{_slugify_token(str(projection_type))}__"
            f"{channel_descriptor}.tif"
        )
        dest_path = dest_dir / dest_name

        tifffile.imwrite(
            dest_path,
            stack,
            photometric="minisblack",
            metadata={"axes": "ZCYX"},
        )

        _validate_stack_channels(dest_path, channel_records)

        merged_rows.append(
            {
                "analysis": analysis,
                "group": group,
                "sample_id": sample_id,
                "projection_type": projection_type,
                "channel_count": len(channel_records),
                "channels": "|".join(channel_labels),
                "export_path": str(dest_path),
                "z_planes": stack.shape[0],
                "height": stack.shape[2],
                "width": stack.shape[3],
                "dtype": str(stack.dtype),
                "source_exports": "|".join(
                    record["export_path"] for record in channel_records
                ),
            }
        )

        print(
            f"[cellpose] Wrote {dest_path} with {len(channel_records)} channels in ZCYX order."
        )

    if not merged_rows:
        print("[cellpose] No multi-channel TIFFs created.")
        return

    csv_path = dest_root / "cellpose_multichannel_metadata.csv"
    fieldnames = [
        "analysis",
        "group",
        "sample_id",
        "projection_type",
        "channel_count",
        "channels",
        "export_path",
        "z_planes",
        "height",
        "width",
        "dtype",
        "source_exports",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[done] Multi-channel Cellpose metadata written to {csv_path}")


def _channel_summary(records: Iterable[dict]) -> str:
    summary = sorted(
        {
            (rec.get("channel_marker") or rec.get("channel_canonical") or rec.get("channel") or "unknown")
            for rec in records
        }
    )
    return ", ".join(summary)


def _validate_stack_channels(path: Path, channel_records: List[dict]) -> None:
    expected = len(channel_records)
    expected_summary = ", ".join(
        _slugify_token(
            record.get("channel_marker")
            or record.get("channel_canonical")
            or record.get("channel")
        )
        for record in channel_records
    )

    try:
        data = np.asarray(tifffile.imread(path))
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[warn] Unable to validate {path}: {exc}")
        return

    if data.ndim == 4:
        observed_channels = data.shape[1]
    elif data.ndim == 3:
        observed_channels = data.shape[0]
    else:
        observed_channels = 0

    if observed_channels != expected:
        print(
            f"[warn] Channel count mismatch for {path}: expected {expected} ({expected_summary}), "
            f"observed {observed_channels} with shape {tuple(data.shape)}."
        )
    else:
        print(
            f"[check] Verified {path.name} with {observed_channels} channel(s): {expected_summary}."
        )


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
    create_multichannel_stacks(output_root, all_records)
    print(f"[done] Export folder ready at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
