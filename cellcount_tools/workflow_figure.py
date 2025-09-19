"""Generate a multi-panel workflow figure from an ND2 imaging field."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, gridspec, patheffects
from matplotlib.patches import FancyArrowPatch
from nd2 import ND2File
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation


# ---------------------------------------------------------------------------
# data helpers


def _load_nd2_projection(nd2_path: Path) -> Tuple[np.ndarray, List[str], float | None]:
    """Return (C, Y, X) array, channel names, and XY pixel size in microns.

    The ND2 reader occasionally returns arrays lacking an explicit Z axis or
    channels collapsed into the last dimension. We normalise the array to
    ``(Z, C, Y, X)`` before the Z projection so downstream code can rely on a
    consistent layout.
    """

    nd2_path = Path(nd2_path)
    with ND2File(nd2_path) as nd2_file:
        data = np.asarray(nd2_file.asarray())
        channels_meta = nd2_file.metadata.channels or []
        voxel_size = None
        try:
            vs = nd2_file.voxel_size()
        except Exception:  # pragma: no cover - rare metadata path
            vs = None
        if vs is not None:
            voxel_size = float(vs.x) if getattr(vs, "x", None) else None

    # Harmonise dimensionality.
    if data.ndim == 4:
        harmonised = data
    elif data.ndim == 3:
        if len(channels_meta) and len(channels_meta) == data.shape[0]:  # (C, Y, X)
            harmonised = data[np.newaxis, ...]
        elif len(channels_meta) and len(channels_meta) == data.shape[-1]:  # (Y, X, C)
            harmonised = np.moveaxis(data, -1, 0)[np.newaxis, ...]
        else:  # assume (Z, Y, X) with a single channel
            harmonised = data[:, np.newaxis, ...]
    elif data.ndim == 2:  # single plane, single channel
        harmonised = data[np.newaxis, np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported ND2 array shape {data.shape}; cannot build projection")

    if harmonised.ndim != 4:
        raise ValueError(f"Could not reshape ND2 data to (Z, C, Y, X); got shape {harmonised.shape}")

    channel_names: List[str] = []
    for channel in channels_meta:
        channel_meta = getattr(channel, "channel", None)
        if channel_meta is not None:
            name = getattr(channel_meta, "name", None)
        else:
            name = getattr(channel, "name", None)
        if not name:
            name = f"Channel{len(channel_names)}"
        channel_names.append(str(name))

    _, channel_count, _, _ = harmonised.shape
    if len(channel_names) != channel_count:
        channel_names = [
            channel_names[idx] if idx < len(channel_names) else f"Channel{idx}"
            for idx in range(channel_count)
        ]

    projection = harmonised.max(axis=0)  # (C, Y, X)
    return projection.astype(np.float32), channel_names, voxel_size


def _normalise_for_display(img: np.ndarray, *, lower_percentile: float = 1.0, upper_percentile: float = 99.8) -> np.ndarray:
    """Scale the image to 0-1 using percentile clipping for display."""

    vmin, vmax = np.percentile(img, [lower_percentile, upper_percentile])
    if vmax <= vmin:
        vmax = img.max()
        vmin = img.min()
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    scaled = (img - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


_DEFAULT_CHANNEL_ALIASES: Dict[str, Tuple[str, str]] = {
    "dapi": ("DAPI", "DAPI"),
    "cy5": ("SOX2", "SOX2 (Cy5)"),
    "sox2": ("SOX2", "SOX2"),
    "dcx": ("DCX", "DCX (Cy5)"),
    "gfp": ("LHX6", "LHX6 (GFP)"),
    "lhx6": ("LHX6", "LHX6"),
    "488": ("LHX6", "LHX6 (488)"),
    "647": ("SOX2", "SOX2 (647)"),
}


def build_alias_table(overrides: Mapping[str, str] | None = None) -> Dict[str, Tuple[str, str]]:
    table = dict(_DEFAULT_CHANNEL_ALIASES)
    if overrides:
        for key, label in overrides.items():
            key_lower = key.lower()
            clean_label = label.strip()
            table[key_lower] = (clean_label, clean_label)
    return table


def _channel_alias(name: str, alias_table: Mapping[str, Tuple[str, str]]) -> Tuple[str, str]:
    lowered = name.lower()
    for key, (short, long) in alias_table.items():
        if key in lowered:
            return short, long
    return name, name


def _panel_channel_sequence(channel_names: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    priority = ["cy5", "gfp", "dapi"]
    for target in priority:
        match = next((name for name in channel_names if target in name.lower()), None)
        if match and match not in ordered:
            ordered.append(match)
    for name in channel_names:
        if name not in ordered:
            ordered.append(name)
    return ordered


PSEUDO_COLORMAPS: Dict[str, mcolors.Colormap] = {
    "dapi": mcolors.LinearSegmentedColormap.from_list(
        "pseudo_dapi", ["black", (0.15, 0.35, 1.0)]
    ),
    "cy5": mcolors.LinearSegmentedColormap.from_list(
        "pseudo_cy5", ["black", (1.0, 0.1, 0.3)]
    ),
    "gfp": mcolors.LinearSegmentedColormap.from_list(
        "pseudo_gfp", ["black", (0.0, 1.0, 0.1)]
    ),
}


def _build_rgb_image(channels: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, str]]:
    """Return an RGB composite and the channel-to-color mapping."""

    def pick_channel(keywords: Iterable[str], fallback: str) -> Tuple[str, np.ndarray]:
        for key in keywords:
            for name, img in channels.items():
                if key in name.lower():
                    return name, img
        return fallback, channels[fallback]

    default_name = next(iter(channels))
    dapi_name, blue = pick_channel(["dapi"], default_name)
    red_name, red = pick_channel(["cy5"], dapi_name)
    green_name, green = pick_channel(["gfp"], dapi_name)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    mapping = {"red": red_name, "green": green_name, "blue": dapi_name}
    return rgb, mapping


# ---------------------------------------------------------------------------
# visual helpers
# ---------------------------------------------------------------------------
# visual helpers


def _add_scale_bar(
    ax: plt.Axes,
    pixel_size_um: float | None,
    image_shape: Tuple[int, ...],
    *,
    bar_um: float = 50.0,
    color: str = "white",
    pad_frac: float = 0.05,
    height_px: int = 4,
) -> None:
    """Draw a horizontal scale bar on ``ax`` in image pixel coordinates."""

    if not image_shape:
        return

    height, width = image_shape[0], image_shape[1]
    if width <= 0 or height <= 0:
        return

    if pixel_size_um and pixel_size_um > 0:
        bar_px = int(round(bar_um / pixel_size_um))
        if bar_px < 1:
            bar_px = 1
        if bar_px > width * 0.5:
            bar_px = int(width * 0.4)
            bar_um = bar_px * pixel_size_um
        label = f"{bar_um:.0f} μm"
    else:
        bar_px = max(1, int(width * 0.2))
        label = f"{bar_px}px"

    x0 = width * pad_frac
    x1 = x0 + bar_px
    y = height * (1 - pad_frac) - height_px

    ax.plot([x0, x1], [y, y], color=color, linewidth=height_px, solid_capstyle="butt")
    text = ax.text(
        (x0 + x1) / 2,
        y - height_px * 1.2,
        label,
        color=color,
        fontsize=8,
        ha="center",
        va="top",
    )
    text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])


# Determine a pseudo colour map for a given channel name
def _pseudo_colormap_for_channel(name: str) -> mcolors.Colormap | None:
    lowered = name.lower()
    for key, cmap in PSEUDO_COLORMAPS.items():
        if key in lowered:
            return cmap
    return None


# ---------------------------------------------------------------------------
# segmentation helpers


@dataclass
class SegmentationOutputs:
    raw: np.ndarray
    smoothed: np.ndarray
    binary: np.ndarray
    cleaned: np.ndarray
    distance: np.ndarray
    markers: np.ndarray
    labels: np.ndarray


def _segment_dapi(
    dapi_img: np.ndarray,
    *,
    smoothing_sigma: float = 1.2,
    otsu_offset: float = 0.0,
    min_size: int = 80,
    min_distance: int = 10,
    peak_footprint: int = 25,
    open_radius: int = 0,
    close_radius: int = 0,
) -> SegmentationOutputs:
    raw = _normalise_for_display(dapi_img)
    smoothed = filters.gaussian(raw, sigma=smoothing_sigma)
    threshold = filters.threshold_otsu(smoothed) + otsu_offset
    binary = smoothed > threshold

    if open_radius and open_radius > 0:
        selem = morphology.disk(open_radius)
        binary = morphology.opening(binary, selem)
    if close_radius and close_radius > 0:
        selem = morphology.disk(close_radius)
        binary = morphology.closing(binary, selem)

    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    cleaned = ndi.binary_fill_holes(cleaned)

    distance = ndi.distance_transform_edt(cleaned)

    footprint = np.ones((peak_footprint, peak_footprint)) if peak_footprint > 0 else None
    peak_mask = feature.peak_local_max(
        distance,
        labels=cleaned,
        footprint=footprint,
        min_distance=min_distance,
        threshold_abs=0.0,
    )
    marker_mask = np.zeros_like(cleaned, dtype=bool)
    if peak_mask.size > 0:
        marker_mask[tuple(peak_mask.T)] = True
    markers, _ = ndi.label(marker_mask)

    labels = segmentation.watershed(-distance, markers, mask=cleaned)

    return SegmentationOutputs(
        raw=raw,
        smoothed=smoothed,
        binary=binary.astype(bool),
        cleaned=cleaned.astype(bool),
        distance=distance,
        markers=markers,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# measurements


def _compute_cell_table(labels: np.ndarray, channel_images: Dict[str, np.ndarray]) -> pd.DataFrame:
    props = measure.regionprops(labels)
    if not props:
        channel_cols = [f"mean_{name}" for name in channel_images.keys()]
        return pd.DataFrame(
            columns=["cell_id", "centroid_y", "centroid_x", "area_px", *channel_cols]
        )

    records = []
    channel_names = list(channel_images.keys())
    for prop in props:
        record = {
            "cell_id": prop.label,
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "area_px": int(prop.area),
        }
        mask = labels == prop.label
        for name in channel_names:
            img = channel_images[name]
            record[f"mean_{name}"] = float(img[mask].mean())
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


def _choose_example_cell(labels: np.ndarray) -> int:
    props = measure.regionprops(labels)
    if not props:
        return 0
    props = sorted(props, key=lambda p: p.area)
    return props[len(props) // 2].label  # median area cell


def _extract_patch(img: np.ndarray, mask: np.ndarray, padding: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty; cannot extract patch")

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0) + 1

    min_row = max(min_row - padding, 0)
    min_col = max(min_col - padding, 0)
    max_row = min(max_row + padding, img.shape[0])
    max_col = min(max_col + padding, img.shape[1])

    return img[min_row:max_row, min_col:max_col], mask[min_row:max_row, min_col:max_col]


# ---------------------------------------------------------------------------
# figure construction


def generate_workflow_figure(
    nd2_path: str | Path,
    *,
    project_root: str | Path | None = None,
    figure_subdir: str = "figures",
    cells_subdir: str = "cells",
    figure_name: str | None = None,
    segmentation_settings: Dict[str, float] | None = None,
    channel_aliases: Mapping[str, str] | None = None,
) -> Path:
    """Create the workflow illustration and persist artefacts inside a project folder.

    Parameters
    ----------
    nd2_path:
        Source ND2 file used for the illustration.
    project_root:
        Destination directory where outputs will be saved.  Defaults to
        ``<nd2_parent>/cellcount_project``.
    figure_subdir / cells_subdir:
        Sub-directories (created if missing) within ``project_root`` where the
        figure PNG and per-cell CSV will be written.
    figure_name:
        Optional override for the figure filename.  Defaults to
        ``<nd2_stem>_workflow.png``.
    segmentation_settings:
        Optional overrides for the DAPI segmentation routine (e.g. ``{"otsu_offset": 0.05}``).
    channel_aliases:
        Optional mapping from channel tokens (e.g. "cy5", "gfp") to biological marker
        names. These augment the default aliases ``Cy5→SOX2`` and ``GFP→LHX6``.
    """

    nd2_path = Path(nd2_path).expanduser().resolve()
    if project_root is None:
        project_root = nd2_path.parent / "cellcount_project"
    project_root = Path(project_root).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    figure_dir = project_root / figure_subdir
    figure_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = project_root / cells_subdir
    cells_dir.mkdir(parents=True, exist_ok=True)

    if figure_name is None:
        figure_name = f"{nd2_path.stem}_workflow.png"
    figure_path = figure_dir / figure_name

    projection, channel_names, pixel_size_um = _load_nd2_projection(nd2_path)
    channel_images = OrderedDict(
        (name, _normalise_for_display(projection[idx])) for idx, name in enumerate(channel_names)
    )
    alias_table = build_alias_table(channel_aliases)
    panel_channels = _panel_channel_sequence(channel_names)

    short_aliases: Dict[str, str] = {}
    long_aliases: Dict[str, str] = {}
    seen_shorts: set[str] = set()
    for name in channel_names:
        short, long = _channel_alias(name, alias_table)
        if short in seen_shorts:
            short, long = name, name
        seen_shorts.add(short)
        short_aliases[name] = short
        long_aliases[name] = long

    dapi_key = next((n for n in channel_names if "dapi" in n.lower()), channel_names[-1])
    seg_settings = segmentation_settings or {}
    seg = _segment_dapi(projection[channel_names.index(dapi_key)], **seg_settings)

    labelled_channels = {name: projection[idx] for idx, name in enumerate(channel_names)}
    cell_table = _compute_cell_table(seg.labels, labelled_channels)
    rename_map = {f"mean_{name}": f"mean_{short_aliases.get(name, name)}" for name in channel_names}
    column_label_map = {
        renamed: long_aliases.get(name, name)
        for name, renamed in ((name, f"mean_{short_aliases.get(name, name)}") for name in channel_names)
    }
    cell_table = cell_table.rename(columns=rename_map)
    if not cell_table.empty:
        cell_table.to_csv(cells_dir / f"{nd2_path.stem}_cells.csv", index=False)

    example_label = _choose_example_cell(seg.labels)
    if example_label == 0:
        raise RuntimeError("No segmented cells found for example panel.")

    example_mask = seg.labels == example_label
    padded_mask = ndi.binary_dilation(example_mask, iterations=8)
    ring_mask = np.logical_and(padded_mask, ~example_mask)

    patch_image, patch_mask = _extract_patch(seg.raw, example_mask)
    if ring_mask.any():
        patch_ring = _extract_patch(seg.raw, ring_mask)[1]
    else:
        patch_ring = np.zeros_like(patch_mask, dtype=bool)

    channel_patches: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
    for name in panel_channels:
        if name not in channel_images:
            continue
        img_patch, mask_patch = _extract_patch(channel_images[name], example_mask)
        channel_patches[name] = (img_patch, mask_patch)
    for name in channel_names:
        if name not in channel_patches and name in channel_images:
            img_patch, mask_patch = _extract_patch(channel_images[name], example_mask)
            channel_patches[name] = (img_patch, mask_patch)

    # ------------------------------------------------------------------ figure layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.1, 0.9, 0.9], width_ratios=[1.05, 1.2, 1.0])
    fig.subplots_adjust(hspace=0.35, wspace=0.32)

    # Panel A ------------------------------------------------------------
    panelA_gs = gs[0, 0].subgridspec(2, 2)
    rgb, rgb_mapping = _build_rgb_image(channel_images)
    red_channel = rgb_mapping.get("red", dapi_key)
    green_channel = rgb_mapping.get("green", dapi_key)

    dapi_cmap = _pseudo_colormap_for_channel(dapi_key) or "gray"
    red_cmap = _pseudo_colormap_for_channel(red_channel) or "gray"
    green_cmap = _pseudo_colormap_for_channel(green_channel) or "gray"

    panelA_defs = [
        (rgb, "Merged RGB", None),
        (
            channel_images.get(dapi_key, next(iter(channel_images.values()))),
            long_aliases.get(dapi_key, "DAPI"),
            dapi_cmap,
        ),
        (
            channel_images.get(red_channel, next(iter(channel_images.values()))),
            long_aliases.get(red_channel, red_channel),
            red_cmap,
        ),
        (
            channel_images.get(green_channel, next(iter(channel_images.values()))),
            long_aliases.get(green_channel, green_channel),
            green_cmap,
        ),
    ]

    panelA_axes = [fig.add_subplot(panelA_gs[i, j]) for i in range(2) for j in range(2)]
    for ax, (img, title, cmap) in zip(panelA_axes, panelA_defs):
        if cmap is None:
            im = ax.imshow(img)
        else:
            im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        _add_scale_bar(ax, pixel_size_um, img.shape)

    _annotate_panel(panelA_axes[0], "A")
    mapping_parts = []
    for color in ("red", "green", "blue"):
        name = rgb_mapping.get(color)
        if not name:
            continue
        mapping_parts.append(f"{long_aliases.get(name, name)} → {color}")
    if mapping_parts:
        panelA_axes[0].text(
            0.02,
            0.05,
            "RGB mapping: " + ", ".join(mapping_parts),
            transform=panelA_axes[0].transAxes,
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.4, "edgecolor": "none"},
        )

    # Panel B ------------------------------------------------------------
    panelB_gs = gs[0, 1:].subgridspec(2, 3)
    b_axes = [fig.add_subplot(panelB_gs[i, j]) for i in range(2) for j in range(3)]
    sigma = seg_settings.get("smoothing_sigma", 1.2)
    otsu_offset = seg_settings.get("otsu_offset", 0.0)
    b_images = [
        (seg.raw, "B1. Raw DAPI", "Nuclear stain"),
        (seg.smoothed, "B2. Gaussian smoothed", f"σ = {sigma:g}"),
        (seg.binary.astype(float), "B3. Otsu threshold", f"Offset Δ={otsu_offset:+.2f}"),
        (seg.cleaned.astype(float), "B4. Mask cleaned", "Remove noise, fill holes"),
        (_normalise_for_display(seg.distance), "B5. Distance + seeds", "Find watershed peaks"),
        (seg.labels, "B6. Labeled nuclei", "Final segmentation"),
    ]

    for ax, (img, title, subtitle) in zip(b_axes, b_images):
        if title.startswith("B5"):
            im = ax.imshow(img, cmap="magma")
            ax.contour(seg.markers > 0, colors="cyan", linewidths=0.6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
        elif title.startswith("B6"):
            ax.imshow(seg.raw, cmap="gray", alpha=0.7)
            im = ax.imshow(seg.labels, cmap="turbo", alpha=0.5)
            for prop in measure.regionprops(seg.labels):
                y, x = prop.centroid
                text = ax.text(
                    x,
                    y,
                    str(prop.label),
                    fontsize=6,
                    color="white",
                    ha="center",
                    va="center",
                )
                text.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="black")])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
        else:
            cmap = "gray" if img.ndim == 2 else "turbo"
            im = ax.imshow(img, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            if "Threshold" in title or "Mask" in title:
                cbar.set_ticks([0, 1])
        ax.set_title(title, fontsize=8, pad=1)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes, fontsize=7, ha="center")
        ax.axis("off")
        _add_scale_bar(ax, pixel_size_um, seg.raw.shape)

    _annotate_panel(b_axes[0], "B")
    b_axes[3].text(0.0, -0.18, "Stepwise segmentation pipeline", transform=b_axes[3].transAxes, fontsize=10)

    # Panel C ------------------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    im_patch = axC.imshow(patch_image, cmap="gray")
    axC.contour(patch_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    if patch_ring.any():
        axC.contour(patch_ring.astype(float), levels=[0.5], colors="orange", linestyles="--", linewidths=1.0)
    axC.set_title("Per-cell measurement region")
    axC.axis("off")
    _annotate_panel(axC, "C")
    cbarC = fig.colorbar(im_patch, ax=axC, fraction=0.046, pad=0.02)
    cbarC.ax.tick_params(labelsize=7)
    _add_scale_bar(axC, pixel_size_um, patch_image.shape)

    # arrows towards channel insets
    inset_positions = []
    for idx, (name, (img, mask)) in enumerate(channel_patches.items()):
        inset = fig.add_axes([
            axC.get_position().x1 + 0.02,
            axC.get_position().y1 - (idx + 1) * 0.18,
            0.12,
            0.16,
        ])
        im_inset = inset.imshow(img, cmap="gray")
        inset.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
        inset.set_title(long_aliases.get(name, name), fontsize=8)
        inset.axis("off")
        cbar_inset = fig.colorbar(im_inset, ax=inset, fraction=0.35, pad=0.02)
        cbar_inset.ax.tick_params(labelsize=6)
        inset_positions.append(inset.get_position())

    for inset_pos in inset_positions:
        arrow = FancyArrowPatch(
            (axC.get_position().x1, axC.get_position().y1 - 0.1),
            (inset_pos.x0, inset_pos.y0 + inset_pos.height / 2),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=15,
            lw=1.5,
            color="white",
        )
        fig.patches.append(arrow)

    measurement_channels = [
        long_aliases[name] for name in panel_channels if name != dapi_key and name in long_aliases
    ]
    if measurement_channels:
        if len(measurement_channels) == 1:
            channel_text = measurement_channels[0]
        elif len(measurement_channels) == 2:
            channel_text = " & ".join(measurement_channels)
        else:
            channel_text = ", ".join(measurement_channels[:2]) + "…"
    else:
        channel_text = "additional markers"
    axC.text(
        0.02,
        -0.12,
        f"Compute mean intensity per channel inside mask;\n{channel_text} shown here.",
        transform=axC.transAxes,
        fontsize=9,
    )

    # Panel D ------------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1:])
    axD.axis("off")
    _annotate_panel(axD, "D")

    display_cols = ["cell_id", "centroid_y", "centroid_x", "area_px"]
    channel_cols = [col for col in cell_table.columns if col.startswith("mean_")]
    display_cols.extend(channel_cols[:3])

    if not cell_table.empty:
        table_data = cell_table[display_cols].head(6).round(1)
        mpl_table = axD.table(
            cellText=table_data.values,
            colLabels=[
                (
                    f"mean {column_label_map.get(col, col.replace('mean_', ''))}"
                    if col.startswith("mean_")
                    else col
                )
                for col in display_cols
            ],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(9)
        mpl_table.scale(1, 1.3)
    else:
        axD.text(0.5, 0.5, "No nuclei detected", ha="center", va="center", fontsize=12)

    axD.set_title("Per-cell measurement table (one row per nucleus)")

    # Panel E ------------------------------------------------------------
    axE = fig.add_subplot(gs[2, 0])
    _annotate_panel(axE, "E")

    non_dapi_channels = [col for col in channel_cols if "dapi" not in col.lower()]
    if not non_dapi_channels:
        non_dapi_channels = channel_cols

    if channel_cols:
        for col in non_dapi_channels[:2]:
            values = cell_table[col]
            label = column_label_map.get(col, col.replace("mean_", ""))
            axE.hist(values, bins=30, alpha=0.5, label=f"Histogram {label}")
            if len(values) > 0:
                values_sorted = np.sort(values)
                ecdf = np.linspace(0, 1, len(values_sorted), endpoint=False)
                axE.plot(values_sorted, ecdf, linewidth=2, label=f"ECDF {label}")

        axE.set_xlabel("Mean intensity (a.u.)")
        axE.set_ylabel("Density / cumulative fraction")
        axE.set_title("Marker distributions across cells")
        axE.legend(fontsize=8)
    else:
        axE.text(0.5, 0.5, "No intensity data", ha="center", va="center", fontsize=12)
        axE.set_axis_off()

    # Panel F ------------------------------------------------------------
    axF = fig.add_subplot(gs[2, 1])
    _annotate_panel(axF, "F")

    if len(channel_cols) >= 2:
        if len(non_dapi_channels) >= 2:
            xcol, ycol = non_dapi_channels[:2]
        else:
            xcol, ycol = channel_cols[:2]

        scatter = axF.scatter(
            cell_table[xcol],
            cell_table[ycol],
            s=15,
            c=cell_table["area_px"],
            cmap="viridis",
            alpha=0.7,
        )
        axF.set_xlabel(f"Mean {column_label_map.get(xcol, xcol.replace('mean_', ''))}")
        axF.set_ylabel(f"Mean {column_label_map.get(ycol, ycol.replace('mean_', ''))}")
        axF.set_title("Marker co-expression per cell")
        cbar_scatter = fig.colorbar(scatter, ax=axF, fraction=0.046, pad=0.02)
        cbar_scatter.ax.set_ylabel("Cell area (px)", fontsize=8)
        cbar_scatter.ax.tick_params(labelsize=7)
    else:
        axF.text(0.5, 0.5, "Need ≥2 channels", ha="center", va="center", fontsize=12)
        axF.set_axis_off()

    # Panel G ------------------------------------------------------------
    axG = fig.add_subplot(gs[2, 2])
    _annotate_panel(axG, "G")
    axG.axis("off")
    axG.set_title("Workflow outputs")
    outputs_text = "\n".join(
        [
            "output/",
            " ├─ masks/ sample_labels.tif",
            " ├─ qc/ sample_mask_build.png",
            " ├─ cells/ sample_cells.csv",
            " ├─ plots/ marker_histograms.png",
            " ├─ plots/ marker_ecdf.png",
            " └─ plots/ marker_scatter.png",
        ]
    )
    axG.text(0.05, 0.9, outputs_text, family="monospace", fontsize=11, verticalalignment="top")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return figure_path


def _annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.1,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
