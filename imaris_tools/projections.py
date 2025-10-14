from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import h5py  # type: ignore

from .metadata import (
    ImarisChannelMetadata,
    ImarisMetadata,
    ImarisProjectionResult,
    _default_color_for_index,
    _discover_channels,
    _read_metadata_from_handle,
)

PathLike = Union[str, Path]


def compute_max_projections(
    path: PathLike,
    *,
    resolution_level: int = 0,
    time_point: int = 0,
) -> Tuple[ImarisMetadata, "OrderedDict[int, np.ndarray]"]:
    """Compute a max projection for each channel found in an Imaris file."""
    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        metadata = _read_metadata_from_handle(
            handle,
            file_path=file_path,
            resolution_level=resolution_level,
            time_point=time_point,
        )
        channel_datasets = sorted(
            _discover_channels(handle, resolution_level=resolution_level, time_point=time_point),
            key=lambda item: item[0],
        )
        projections: "OrderedDict[int, np.ndarray]" = OrderedDict()
        for index, dataset in channel_datasets:
            projections[index] = _max_intensity_projection(dataset)
    return metadata, projections


def process_file(
    path: PathLike,
    *,
    resolution_level: int = 0,
    time_point: int = 0,
    composite_dtype: Union[np.dtype, type] = np.uint16,
) -> ImarisProjectionResult:
    """Generate channel projections and an RGB composite for a single file."""
    metadata, projections_by_index = compute_max_projections(
        path,
        resolution_level=resolution_level,
        time_point=time_point,
    )

    channel_images: Dict[str, np.ndarray] = OrderedDict()
    channel_names: Dict[int, str] = {}
    used_names: Dict[str, int] = {}
    for channel in sorted(metadata.channels, key=lambda c: c.index):
        array = projections_by_index.get(channel.index)
        if array is None:
            continue
        name = _ensure_unique_name(channel.name, used_names)
        channel_images[name] = array
        channel_names[channel.index] = name

    # Include channels that were not described in the XML metadata.
    if len(channel_images) < len(projections_by_index):
        missing_indices = set(projections_by_index.keys()) - {
            channel.index for channel in metadata.channels
        }
        for idx in sorted(missing_indices):
            name = _ensure_unique_name(f"Channel {idx}", used_names)
            channel_images[name] = projections_by_index[idx]
            channel_names[idx] = name

    composite = _compose_rgb(
        projections_by_index,
        channel_metadata=metadata.channels,
        dtype=composite_dtype,
    )

    return ImarisProjectionResult(
        source_path=Path(path),
        metadata=metadata,
        channel_projections=channel_images,
        channel_names=channel_names,
        composite_rgb=composite,
    )


def process_directory(
    folder: PathLike,
    *,
    pattern: str = "*.ims",
    recursive: bool = False,
    resolution_level: int = 0,
    time_point: int = 0,
    composite_dtype: Union[np.dtype, type] = np.uint16,
) -> List[ImarisProjectionResult]:
    """Process every ``.ims`` file in ``folder`` matching ``pattern``."""
    directory = Path(folder)
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")

    iterator: Iterable[Path]
    if recursive:
        iterator = sorted(directory.rglob(pattern))
    else:
        iterator = sorted(directory.glob(pattern))

    results: List[ImarisProjectionResult] = []
    for path in iterator:
        if not path.is_file():
            continue
        results.append(
            process_file(
                path,
                resolution_level=resolution_level,
                time_point=time_point,
                composite_dtype=composite_dtype,
            )
        )
    return results


def _max_intensity_projection(dataset: h5py.Dataset) -> np.ndarray:
    """Compute the max projection along the first axis of ``dataset``."""
    if dataset.ndim < 3:
        data = dataset[()]
        if data.ndim == 2:
            return np.array(data)
        raise ValueError("Expected a (Z, Y, X) dataset for channel projection")

    proj = dataset[0, ...].astype(dataset.dtype, copy=True)
    for z_index in range(1, dataset.shape[0]):
        slice_data = dataset[z_index, ...]
        np.maximum(proj, slice_data, out=proj)
    return np.array(proj)


def _compose_rgb(
    projections_by_index: "OrderedDict[int, np.ndarray]",
    *,
    channel_metadata: Sequence[ImarisChannelMetadata],
    dtype: Union[np.dtype, type],
) -> np.ndarray:
    if not projections_by_index:
        target_dtype = np.dtype(dtype)
        return np.zeros((0, 0, 3), dtype=target_dtype)

    first = next(iter(projections_by_index.values()))
    height, width = first.shape[-2], first.shape[-1]
    composite = np.zeros((height, width, 3), dtype=np.float32)

    # Merge channels described in metadata first.
    for channel in sorted(channel_metadata, key=lambda c: c.index):
        array = projections_by_index.get(channel.index)
        if array is None:
            continue
        composite += _projection_to_rgb(array, channel.color_rgb)

    # Include any channels not described in the metadata.
    known_indices = {channel.index for channel in channel_metadata}
    for index, array in projections_by_index.items():
        if index in known_indices:
            continue
        composite += _projection_to_rgb(array, _default_color_for_index(index))

    return _finalize_rgb(composite, dtype=dtype)


def _projection_to_rgb(array: np.ndarray, color_rgb: Tuple[float, float, float]) -> np.ndarray:
    normalized = _normalize_projection(array)
    r, g, b = color_rgb
    rgb = np.zeros((*normalized.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized * r
    rgb[..., 1] = normalized * g
    rgb[..., 2] = normalized * b
    return rgb


def _normalize_projection(array: np.ndarray) -> np.ndarray:
    data = array.astype(np.float32, copy=False)
    max_val = float(np.max(data))
    if max_val == 0:
        return np.zeros_like(data)
    return data / max_val


def _finalize_rgb(composite: np.ndarray, *, dtype: Union[np.dtype, type]) -> np.ndarray:
    composite = np.clip(composite, 0.0, 1.0)
    target_dtype = np.dtype(dtype)
    if target_dtype == np.float32 or target_dtype == np.float64:
        return composite.astype(target_dtype, copy=False)
    if target_dtype == np.uint8:
        return np.round(composite * 255.0).astype(np.uint8)
    if target_dtype == np.uint16:
        return np.round(composite * 65535.0).astype(np.uint16)
    return composite.astype(target_dtype)


def colorize_projection(
    array: np.ndarray,
    color_rgb: Tuple[float, float, float],
    *,
    dtype: Union[np.dtype, type] = np.uint16,
) -> np.ndarray:
    """Return a colourized RGB view of ``array`` using ``color_rgb`` weights."""
    rgb = _projection_to_rgb(array, color_rgb)
    return _finalize_rgb(rgb, dtype=dtype)


def _ensure_unique_name(name: str, used: Dict[str, int]) -> str:
    base = name or "Channel"
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count}"


__all__ = [
    "compute_max_projections",
    "process_file",
    "process_directory",
    "colorize_projection",
]
