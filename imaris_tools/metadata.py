from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import h5py  # type: ignore

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ImarisChannelMetadata:
    """Metadata describing a single Imaris channel."""

    index: int
    name: str
    color_rgb: Tuple[float, float, float]
    excitation_wavelength_nm: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    detection_wavelength_nm: Optional[float] = None


@dataclass(frozen=True)
class ImarisMetadata:
    """Parsed metadata for an Imaris dataset."""

    source_path: Path
    name: Optional[str]
    description: Optional[str]
    dimensions_xyzct: Tuple[int, int, int, int, int]
    voxel_size_um: Optional[Tuple[float, float, float]]
    time_delta_s: Optional[float]
    channels: Tuple[ImarisChannelMetadata, ...]
    raw_xml: Optional[str] = None


@dataclass(frozen=True)
class ImarisProjectionResult:
    """Container returned when processing an Imaris file."""

    source_path: Path
    metadata: ImarisMetadata
    channel_projections: Dict[str, np.ndarray]
    composite_rgb: np.ndarray


def read_metadata(
    path: PathLike,
    *,
    resolution_level: int = 0,
    time_point: int = 0,
) -> ImarisMetadata:
    """Read high-level metadata from an Imaris ``.ims`` file.

    Parameters
    ----------
    path:
        Path to the ``.ims`` file.
    resolution_level:
        Resolution level to inspect while inferring dataset dimensions.
    time_point:
        Time point index to inspect while inferring dataset dimensions.
    """
    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        return _read_metadata_from_handle(
            handle,
            file_path=file_path,
            resolution_level=resolution_level,
            time_point=time_point,
        )


def _read_metadata_from_handle(
    handle: h5py.File,
    *,
    file_path: Path,
    resolution_level: int,
    time_point: int,
) -> ImarisMetadata:
    raw_xml = _read_dataset_info_xml(handle)
    xml_root = ET.fromstring(raw_xml) if raw_xml else None
    channel_entries = _parse_channels(xml_root) if xml_root is not None else {}

    channel_datasets = list(
        _discover_channels(handle, resolution_level=resolution_level, time_point=time_point)
    )
    channel_indices = [index for index, _ in channel_datasets]
    channel_indices.sort()

    channels: List[ImarisChannelMetadata] = []
    for idx in channel_indices:
        entry = channel_entries.get(idx)
        if entry is None:
            entry = ImarisChannelMetadata(
                index=idx,
                name=f"Channel {idx}",
                color_rgb=_default_color_for_index(idx),
            )
        channels.append(entry)

    dimensions = _infer_dimensions(
        xml_root=xml_root,
        channel_datasets=channel_datasets,
        channel_count=len(channel_datasets) or len(channel_entries),
        time_points=_discover_time_point_count(handle, resolution_level=resolution_level),
    )
    voxel_size = _infer_voxel_size(xml_root, dimensions)
    time_delta = _infer_time_delta(xml_root)

    name = xml_root.findtext(".//Image/Name") if xml_root is not None else None
    description = xml_root.findtext(".//Image/Description") if xml_root is not None else None

    if not channels and channel_entries:
        # Include metadata-only channels to avoid losing information.
        channels = sorted(channel_entries.values(), key=lambda c: c.index)

    return ImarisMetadata(
        source_path=file_path,
        name=name,
        description=description,
        dimensions_xyzct=dimensions,
        voxel_size_um=voxel_size,
        time_delta_s=time_delta,
        channels=tuple(channels),
        raw_xml=raw_xml,
    )


def _read_dataset_info_xml(handle: h5py.File) -> Optional[str]:
    if "DataSetInfo" not in handle:
        return None
    dataset = handle["DataSetInfo"]
    raw = dataset[()]
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    if isinstance(raw, (bytes, bytearray, memoryview)):
        data = bytes(raw)
    else:
        data = str(raw).encode("utf-8")
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def _parse_channels(root: Optional[ET.Element]) -> Dict[int, ImarisChannelMetadata]:
    if root is None:
        return {}
    channels: Dict[int, ImarisChannelMetadata] = {}
    for idx, node in enumerate(root.findall(".//Channels/Channel")):
        index = _parse_int(node.get("Id")) or _parse_int(node.get("Index")) or idx
        name = (node.findtext("Name") or f"Channel {index}").strip()
        color_node = node.find("Color")
        if color_node is not None:
            color = tuple(
                _clip_unit_float(_normalize_color_component(color_node.findtext(component)))
                for component in ("Red", "Green", "Blue")
            )
        else:
            color = _default_color_for_index(index)
        channels[index] = ImarisChannelMetadata(
            index=index,
            name=name,
            color_rgb=color,
            excitation_wavelength_nm=_parse_float(node.findtext("ExcitationWavelength")),
            emission_wavelength_nm=_parse_float(node.findtext("EmissionWavelength")),
            detection_wavelength_nm=_parse_float(node.findtext("DetectionWavelength")),
        )
    return channels


def _infer_dimensions(
    *,
    xml_root: Optional[ET.Element],
    channel_datasets: Sequence[Tuple[int, h5py.Dataset]],
    channel_count: int,
    time_points: int,
) -> Tuple[int, int, int, int, int]:
    x = y = z = 0
    if xml_root is not None:
        x = _parse_int(xml_root.findtext(".//Image/Dimensions/X")) or 0
        y = _parse_int(xml_root.findtext(".//Image/Dimensions/Y")) or 0
        z = _parse_int(xml_root.findtext(".//Image/Dimensions/Z")) or 0
        if channel_count == 0:
            channel_count = _parse_int(xml_root.findtext(".//Image/Dimensions/C")) or 0
        if time_points == 0:
            time_points = _parse_int(xml_root.findtext(".//Image/Dimensions/T")) or 0

    if channel_datasets:
        sample_shape = channel_datasets[0][1].shape
        if len(sample_shape) >= 3:
            z = z or int(sample_shape[-3])
            y = y or int(sample_shape[-2])
            x = x or int(sample_shape[-1])

    return (x, y, z, channel_count, time_points)


def _infer_voxel_size(
    xml_root: Optional[ET.Element],
    dimensions_xyzct: Tuple[int, int, int, int, int],
) -> Optional[Tuple[float, float, float]]:
    if xml_root is None:
        return None

    voxel = (
        _parse_float(xml_root.findtext(".//Image/VoxelSize/X")),
        _parse_float(xml_root.findtext(".//Image/VoxelSize/Y")),
        _parse_float(xml_root.findtext(".//Image/VoxelSize/Z")),
    )
    if all(v is not None for v in voxel):
        return (voxel[0] or 0.0, voxel[1] or 0.0, voxel[2] or 0.0)

    extent_min = _parse_vector3(xml_root.findtext(".//Image/ExtMin") or "")
    extent_max = _parse_vector3(xml_root.findtext(".//Image/ExtMax") or "")
    x, y, z, _, _ = dimensions_xyzct
    if extent_min and extent_max and x > 1 and y > 1 and z > 1:
        spacing = tuple(
            (max_val - min_val) / float(size - 1)
            for (min_val, max_val, size) in zip(extent_min, extent_max, (x, y, z))
        )
        return spacing  # type: ignore[return-value]
    return None


def _infer_time_delta(xml_root: Optional[ET.Element]) -> Optional[float]:
    if xml_root is None:
        return None
    delta = _parse_float(xml_root.findtext(".//TimeInfo/DeltaT"))
    if delta is not None:
        return delta
    # Some files encode time per frame as a space-separated list.
    times = _parse_vector(xml_root.findtext(".//TimeInfo/TimePoint") or "")
    if len(times) >= 2:
        return times[1] - times[0]
    return None


def _discover_channels(
    handle: h5py.File,
    *,
    resolution_level: int = 0,
    time_point: int = 0,
) -> Iterator[Tuple[int, h5py.Dataset]]:
    group_path = f"DataSet/ResolutionLevel {resolution_level}/TimePoint {time_point}"
    if group_path not in handle:
        return
    time_group = handle[group_path]
    for key, value in time_group.items():
        if not key.lower().startswith("channel"):
            continue
        parts = key.split()
        try:
            index = int(parts[-1])
        except (IndexError, ValueError):
            continue
        if "Data" in value:
            yield index, value["Data"]


def _discover_time_point_count(handle: h5py.File, *, resolution_level: int = 0) -> int:
    group_path = f"DataSet/ResolutionLevel {resolution_level}"
    if group_path not in handle:
        return 0
    res_group = handle[group_path]
    time_indices = []
    for key in res_group.keys():
        if not key.lower().startswith("timepoint"):
            continue
        parts = key.split()
        try:
            time_indices.append(int(parts[-1]))
        except (IndexError, ValueError):
            continue
    if not time_indices:
        return 0
    return max(time_indices) + 1


def _parse_int(value: Optional[Union[str, int]]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(value: Optional[Union[str, float]], default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _clip_unit_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _default_color_for_index(index: int) -> Tuple[float, float, float]:
    palette = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
    ]
    return palette[index % len(palette)]


def _parse_vector(text: str) -> List[float]:
    values: List[float] = []
    for part in text.replace(",", " ").split():
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _parse_vector3(text: str) -> Optional[Tuple[float, float, float]]:
    values = _parse_vector(text)
    if len(values) >= 3:
        return (values[0], values[1], values[2])
    return None


def _normalize_color_component(text: Optional[str]) -> Optional[float]:
    value = _parse_float(text)
    if value is None:
        return None
    if value > 1.0:
        return value / 255.0
    return value


__all__ = [
    "ImarisChannelMetadata",
    "ImarisMetadata",
    "ImarisProjectionResult",
    "read_metadata",
]
