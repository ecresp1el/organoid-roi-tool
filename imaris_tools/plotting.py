from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # pragma: no cover - enforce non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .metadata import ImarisChannelMetadata, default_channel_color
from .projections import process_directory

PathLike = Union[str, Path]


def plot_folder_projection_grid(
    source: PathLike,
    output_pdf: PathLike,
    *,
    pattern: str = "*.ims",
    recursive: bool = False,
    resolution_level: int = 0,
    time_point: int = 0,
    percentile: float = 95.0,
    dpi: int = 150,
) -> Path:
    """
    Generate an N×M grid of max-projection images summarising all datasets in ``source``.

    Rows correspond to individual Imaris files, columns correspond to channels.  Each
    panel displays raw max-projection pixels scaled by the specified percentile and
    annotated with a per-panel colourbar.  The resulting figure is saved as ``output_pdf``.
    """
    folder = Path(source)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist")

    results = process_directory(
        folder,
        pattern=pattern,
        recursive=recursive,
        resolution_level=resolution_level,
        time_point=time_point,
    )
    if not results:
        raise ValueError(f"No .ims files matching '{pattern}' were found in {folder}")

    channel_indices = _collect_channel_indices(results)
    figure = _create_grid_figure(
        results,
        channel_indices,
        percentile=percentile,
    )

    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return pdf_path


def _collect_channel_indices(results) -> List[int]:
    indices: set[int] = set()
    for result in results:
        indices.update(result.channel_names.keys())
    return sorted(indices)


def _create_grid_figure(results, channel_indices: Sequence[int], *, percentile: float) -> plt.Figure:
    row_count = len(results)
    col_count = len(channel_indices)

    fig_width = max(6, col_count * 4)
    fig_height = max(6, row_count * 4)

    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    for row, result in enumerate(results):
        file_label = result.source_path.stem
        for col, channel_index in enumerate(channel_indices):
            ax = axes[row][col]
            _populate_axis(
                ax,
                result,
                channel_index,
                percentile=percentile,
                row_label=file_label if col == 0 else None,
            )

    return fig


def _populate_axis(
    ax: plt.Axes,
    result,
    channel_index: int,
    *,
    percentile: float,
    row_label: Optional[str],
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])

    unique_name = result.channel_names.get(channel_index)
    array = None
    if unique_name is not None:
        array = result.channel_projections.get(unique_name)

    if array is None:
        ax.set_facecolor("lightgrey")
        ax.set_title(f"Channel {channel_index} (missing)", fontsize=10)
        if row_label:
            ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=10)
        return

    vmax = _percentile_or_max(array, percentile)
    cmap = _colormap_for_channel(result.metadata.channels, channel_index)
    img = ax.imshow(array, cmap=cmap, vmin=0, vmax=vmax)
    cb = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel("Intensity", fontsize=8)

    channel_meta = _channel_metadata_lookup(result.metadata.channels, channel_index)
    title = channel_meta.name if channel_meta is not None else unique_name
    ax.set_title(f"{title} (p{percentile:.0f}={vmax:.1f})", fontsize=10)

    if row_label:
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=10)


def _percentile_or_max(array: np.ndarray, percentile: float) -> float:
    positive = array[array > 0]
    if positive.size == 0:
        vmax = float(np.max(array) if array.size else 0.0)
    else:
        vmax = float(np.percentile(positive, percentile))
    return vmax if vmax > 0 else 1.0


def _colormap_for_channel(channels: Sequence[ImarisChannelMetadata], index: int) -> LinearSegmentedColormap:
    entry = _channel_metadata_lookup(channels, index)
    rgb = entry.color_rgb if entry is not None else default_channel_color(index)
    rgb = tuple(float(max(0.0, min(1.0, component))) for component in rgb)
    return LinearSegmentedColormap.from_list(
        f"channel_{index}",
        [(0.0, 0.0, 0.0), rgb],
    )


def _channel_metadata_lookup(
    channels: Sequence[ImarisChannelMetadata],
    index: int,
) -> Optional[ImarisChannelMetadata]:
    for channel in channels:
        if channel.index == index:
            return channel
    return None


__all__ = ["plot_folder_projection_grid"]
