from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import tifffile as tiff  # type: ignore

from .metadata import ImarisChannelMetadata
from .projections import compute_max_projections

PathLike = Union[str, Path]


def save_fluorescent_max_projections(
    source: PathLike,
    *,
    output_root: Optional[PathLike] = None,
    pattern: str = "*.ims",
    recursive: bool = False,
    resolution_level: int = 0,
    time_point: int = 0,
) -> Path:
    """
    Quickly dump max-projection TIFFs for channels that report fluorescence metadata.

    Fluorescent channels are heuristically identified as those with either an
    excitation or emission wavelength present in the Imaris metadata.  If no such
    channels are found, all channels will be exported for that file.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"{source_path} does not exist")

    destination = (
        Path(output_root)
        if output_root is not None
        else (source_path / "fluorescent_max_projections")
    )
    destination.mkdir(parents=True, exist_ok=True)

    files = _collect_files(source_path, pattern=pattern, recursive=recursive)
    for ims_path in files:
        metadata, projections = compute_max_projections(
            ims_path,
            resolution_level=resolution_level,
            time_point=time_point,
        )
        channel_lookup = {channel.index: channel for channel in metadata.channels}

        fluorescent_indices = _fluorescent_indices(metadata.channels)
        if not fluorescent_indices:
            fluorescent_indices = list(projections.keys())

        for index in fluorescent_indices:
            if index not in projections:
                continue
            array = projections[index]
            channel = channel_lookup.get(index)
            basename = _derive_filename(ims_path, index, channel)
            output_path = destination / basename
            tiff.imwrite(output_path, np.asarray(array))

    return destination


def _collect_files(folder: Path, *, pattern: str, recursive: bool) -> List[Path]:
    iterator: Iterable[Path]
    if recursive:
        iterator = folder.rglob(pattern)
    else:
        iterator = folder.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def _fluorescent_indices(channels: Iterable[ImarisChannelMetadata]) -> List[int]:
    indices: List[int] = []
    for channel in channels:
        if (channel.excitation_wavelength_nm is not None) or (channel.emission_wavelength_nm is not None):
            indices.append(channel.index)
    return indices


def _derive_filename(path: Path, index: int, channel: Optional[ImarisChannelMetadata]) -> str:
    base = path.stem
    suffix = f"ch{index}"
    if channel is not None:
        suffix += f"_{_sanitize_name(channel.name)}"
    return f"{base}_{suffix}.tif"


def _sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    cleaned = cleaned.strip("_.")
    return cleaned or "channel"


__all__ = ["save_fluorescent_max_projections"]
