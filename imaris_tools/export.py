from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tifffile as tiff  # type: ignore

from .metadata import (
    ImarisChannelMetadata,
    ImarisMetadata,
    default_channel_color,
)
from .projections import colorize_projection, process_file

PathLike = Union[str, Path]
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def export_directory(
    source: PathLike,
    *,
    output_root: Optional[PathLike] = None,
    csv_path: Optional[PathLike] = None,
    pattern: str = "*.ims",
    recursive: bool = False,
    resolution_level: int = 0,
    time_point: int = 0,
    composite_dtype: Union[np.dtype, type] = np.uint16,
    colorized_dtype: Optional[Union[np.dtype, type]] = None,
    overwrite: bool = False,
) -> Path:
    """Export metadata and projections for every Imaris file in ``source``."""
    base_dir = Path(source)
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} does not exist")

    output_dir = Path(output_root) if output_root is not None else (base_dir / "imaris_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = Path(csv_path) if csv_path is not None else (output_dir / "metadata_summary.csv")
    color_dtype = colorized_dtype if colorized_dtype is not None else composite_dtype

    rows: List[Dict[str, object]] = []
    for ims_path in _iter_ims_files(base_dir, pattern=pattern, recursive=recursive):
        result = process_file(
            ims_path,
            resolution_level=resolution_level,
            time_point=time_point,
            composite_dtype=composite_dtype,
        )

        per_file_dir = output_dir / ims_path.stem
        if per_file_dir.exists() and overwrite:
            shutil.rmtree(per_file_dir)
        per_file_dir.mkdir(parents=True, exist_ok=True)

        metadata_json_path, metadata_xml_path = _write_metadata_files(
            result.metadata,
            channel_names=result.channel_names,
            folder=per_file_dir,
        )

        composite_path = per_file_dir / "composite_rgb.tif"
        tiff.imwrite(composite_path, result.composite_rgb, photometric="rgb")

        channel_lookup: Dict[int, ImarisChannelMetadata] = {
            channel.index: channel for channel in result.metadata.channels
        }

        for index in sorted(result.channel_names.keys()):
            unique_name = result.channel_names[index]
            channel_array = result.channel_projections[unique_name]
            sanitized = _sanitize_name(unique_name)

            grayscale_path = per_file_dir / f"{index:02d}_{sanitized}.tif"
            tiff.imwrite(grayscale_path, channel_array)

            channel_info = channel_lookup.get(index)
            color_rgb = channel_info.color_rgb if channel_info is not None else default_channel_color(index)
            colorized_path = per_file_dir / f"{index:02d}_{sanitized}_color.tif"
            colorized = colorize_projection(channel_array, color_rgb, dtype=color_dtype)
            tiff.imwrite(colorized_path, colorized, photometric="rgb")

            channel_shape = result.metadata.channel_shapes.get(index)
            z_planes = channel_shape[0] if channel_shape else result.metadata.dimensions_xyzct[2]
            height, width = channel_array.shape[-2], channel_array.shape[-1]
            voxel = result.metadata.voxel_size_um or (None, None, None)

            rows.append(
                {
                    "source_path": str(ims_path),
                    "file_name": ims_path.name,
                    "output_folder": str(per_file_dir),
                    "channel_index": index,
                    "channel_name": channel_info.name if channel_info is not None else unique_name,
                    "channel_unique_name": unique_name,
                    "channel_color_r": color_rgb[0],
                    "channel_color_g": color_rgb[1],
                    "channel_color_b": color_rgb[2],
                    "channel_z_planes": z_planes,
                    "width_px": width,
                    "height_px": height,
                    "projection_dtype": str(channel_array.dtype),
                    "colorized_dtype": str(colorized.dtype),
                    "voxel_size_x_um": voxel[0],
                    "voxel_size_y_um": voxel[1],
                    "voxel_size_z_um": voxel[2],
                    "time_delta_s": result.metadata.time_delta_s,
                    "time_points": result.metadata.dimensions_xyzct[4],
                    "resolution_level": resolution_level,
                    "time_point": time_point,
                    "projection_path": str(grayscale_path),
                    "colorized_path": str(colorized_path),
                    "composite_path": str(composite_path),
                    "metadata_json_path": str(metadata_json_path),
                    "metadata_xml_path": str(metadata_xml_path) if metadata_xml_path else "",
                }
            )

    _write_metadata_csv(metadata_csv, rows)
    return metadata_csv


def _iter_ims_files(directory: Path, *, pattern: str, recursive: bool) -> Iterable[Path]:
    iterator: Iterable[Path]
    if recursive:
        iterator = directory.rglob(pattern)
    else:
        iterator = directory.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def _sanitize_name(name: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", name.strip())
    sanitized = sanitized.strip("_.")
    return sanitized or "channel"


def _write_metadata_files(
    metadata: ImarisMetadata,
    *,
    channel_names: Dict[int, str],
    folder: Path,
) -> Tuple[Path, Optional[Path]]:
    metadata_json_path = folder / "metadata.json"
    dataset_info_path: Optional[Path] = None
    if metadata.raw_xml:
        dataset_info_path = folder / "dataset_info.xml"
        dataset_info_path.write_text(metadata.raw_xml)
    _write_metadata_json(metadata, channel_names=channel_names, json_path=metadata_json_path, xml_path=dataset_info_path)
    return metadata_json_path, dataset_info_path


def _write_metadata_json(
    metadata: ImarisMetadata,
    *,
    channel_names: Dict[int, str],
    json_path: Path,
    xml_path: Optional[Path],
) -> None:
    channel_lookup: Dict[int, ImarisChannelMetadata] = {
        channel.index: channel for channel in metadata.channels
    }
    payload = {
        "source_path": str(metadata.source_path),
        "name": metadata.name,
        "description": metadata.description,
        "dimensions_xyzct": list(metadata.dimensions_xyzct),
        "voxel_size_um": list(metadata.voxel_size_um) if metadata.voxel_size_um else None,
        "time_delta_s": metadata.time_delta_s,
        "channel_shapes": {str(index): list(shape) for index, shape in metadata.channel_shapes.items()},
        "channels": [],
        "dataset_info_xml": str(xml_path) if xml_path else None,
    }
    for index, unique_name in sorted(channel_names.items()):
        channel_info = channel_lookup.get(index)
        entry = {
            "index": index,
            "unique_name": unique_name,
            "shape": list(metadata.channel_shapes.get(index, ())),
            "metadata_defined": channel_info is not None,
            "color_rgb": (
                list(channel_info.color_rgb)
                if channel_info is not None
                else list(default_channel_color(index))
            ),
        }
        if channel_info is not None:
            entry.update(
                {
                    "name": channel_info.name,
                    "excitation_wavelength_nm": channel_info.excitation_wavelength_nm,
                    "emission_wavelength_nm": channel_info.emission_wavelength_nm,
                    "detection_wavelength_nm": channel_info.detection_wavelength_nm,
                }
            )
        payload["channels"].append(entry)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_metadata_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        # Create empty CSV with header for consistency.
        fieldnames = [
            "source_path",
            "file_name",
            "output_folder",
            "channel_index",
            "channel_name",
            "channel_unique_name",
            "channel_color_r",
            "channel_color_g",
            "channel_color_b",
            "channel_z_planes",
            "width_px",
            "height_px",
            "projection_dtype",
            "colorized_dtype",
            "voxel_size_x_um",
            "voxel_size_y_um",
            "voxel_size_z_um",
            "time_delta_s",
            "time_points",
            "resolution_level",
            "time_point",
            "projection_path",
            "colorized_path",
            "composite_path",
            "metadata_json_path",
            "metadata_xml_path",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = ["export_directory"]
