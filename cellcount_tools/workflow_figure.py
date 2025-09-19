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

from .project_config import load_channel_aliases


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
    """Return an RGB composite and the channel-to-color mapping.

    Only channels that exist contribute; missing channels yield zero-filled planes so the
    merged view does not fabricate pseudo-colours.
    """

    def pick_channel(keywords: Iterable[str]) -> Tuple[str | None, np.ndarray | None]:
        for key in keywords:
            for name, img in channels.items():
                if key in name.lower():
                    return name, img
        return None, None

    # DAPI (blue) is mandatory for the merge; if absent, take the first channel.
    default_name = next(iter(channels))
    dapi_name, blue = pick_channel(["dapi"])
    if blue is None:
        dapi_name = default_name
        blue = channels[dapi_name]

    red_name, red = pick_channel(["cy5", "dcx", "sox2", "647"])
    green_name, green = pick_channel(["gfp", "lhx6", "488"])

    shape = blue.shape
    zeros = np.zeros(shape, dtype=blue.dtype)
    if red is None:
        red = zeros
    if green is None:
        green = zeros

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    mapping: Dict[str, str] = {"blue": dapi_name}
    if red_name:
        mapping["red"] = red_name
    if green_name:
        mapping["green"] = green_name
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
    tophat_radius: int = 0
    max_size: int | None = None


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
    tophat_radius: int = 0,
    max_size: int | None = None,
) -> SegmentationOutputs:
    working = dapi_img.astype(np.float32, copy=True)
    if tophat_radius and tophat_radius > 0:
        selem = morphology.disk(tophat_radius)
        working = morphology.white_tophat(working, footprint=selem)
    raw = _normalise_for_display(working)
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

    if max_size is not None and max_size > 0:
        for region in measure.regionprops(labels):
            if region.area > max_size:
                labels[labels == region.label] = 0

    return SegmentationOutputs(
        raw=raw,
        smoothed=smoothed,
        binary=binary.astype(bool),
        cleaned=cleaned.astype(bool),
        distance=distance,
        markers=markers,
        labels=labels,
        tophat_radius=int(tophat_radius or 0),
        max_size=max_size,
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
    project_root_path = project_root
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

    project_aliases = load_channel_aliases(project_root_path)
    alias_overrides = dict(project_aliases)
    if channel_aliases:
        alias_overrides.update({k.lower(): v.strip() for k, v in channel_aliases.items()})
    alias_table = build_alias_table(alias_overrides or None)
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
    FIG_W, FIG_H = 1920, 1164
    dpi = 100
    fig = plt.figure(figsize=(FIG_W / dpi, FIG_H / dpi), dpi=dpi)

    def add_axes_pixels(x: float, y: float, width: float, height: float) -> plt.Axes:
        return fig.add_axes([x / FIG_W, y / FIG_H, width / FIG_W, height / FIG_H])

    def add_caption(x: float, y: float, width: float, text: str) -> None:
        fig.text((x + width / 2) / FIG_W, (y + CAPTION / 2) / FIG_H, text, fontsize=18, fontweight="bold", ha="center", va="center")

    def add_panel_label(ax: plt.Axes, label: str) -> None:
        ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=18, fontweight="bold", ha="right", va="bottom")

    MARGIN_LEFT = 48
    MARGIN_TOP = 48
    ROW_GAP = 24
    CAPTION = 32
    ROW1_H = 420
    ROW2_H = 300
    ROW3_H = 300

    row1_y = FIG_H - MARGIN_TOP - ROW1_H
    row2_y = row1_y - ROW_GAP - ROW2_H
    row3_y = row2_y - ROW_GAP - ROW3_H

    cbar_w = 10
    cbar_pad = 6
    sub_gap = 12

    # Panel A ------------------------------------------------------------
    panel_a = dict(x=MARGIN_LEFT, y=row1_y, w=540, h=ROW1_H)
    panel_a_inner_h = panel_a["h"] - CAPTION
    panel_a_inner_y = panel_a["y"] + CAPTION
    slot_w = (panel_a["w"] - sub_gap) / 2
    slot_h = (panel_a_inner_h - sub_gap) / 2

    rgb, rgb_mapping = _build_rgb_image(channel_images)
    red_channel = rgb_mapping.get("red", dapi_key)
    green_channel = rgb_mapping.get("green", dapi_key)

    dapi_cmap = _pseudo_colormap_for_channel(dapi_key) or "gray"
    red_cmap = _pseudo_colormap_for_channel(red_channel) or "gray"
    green_cmap = _pseudo_colormap_for_channel(green_channel) or "gray"

    panelA_defs = [
        (rgb, "Merged RGB", None),
        (channel_images.get(dapi_key, next(iter(channel_images.values()))), long_aliases.get(dapi_key, "DAPI"), dapi_cmap),
        (channel_images.get(red_channel, next(iter(channel_images.values()))), long_aliases.get(red_channel, red_channel), red_cmap),
        (channel_images.get(green_channel, next(iter(channel_images.values()))), long_aliases.get(green_channel, green_channel), green_cmap),
    ]

    panelA_axes: List[plt.Axes] = []
    for idx, (img, title, cmap) in enumerate(panelA_defs):
        row = idx // 2
        col = idx % 2
        slot_x = panel_a["x"] + col * (slot_w + sub_gap)
        slot_y = panel_a_inner_y + (1 - row) * (slot_h + sub_gap)
        has_cbar = cmap is not None
        img_w = slot_w - (cbar_w + cbar_pad) if has_cbar else slot_w
        ax = add_axes_pixels(slot_x, slot_y, img_w, slot_h)
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.axis("off")
        _add_scale_bar(ax, pixel_size_um, img.shape)
        ax.set_title(title, fontsize=9, pad=4)
        if has_cbar:
            cax = add_axes_pixels(slot_x + img_w + cbar_pad, slot_y, cbar_w, slot_h)
            norm = mcolors.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, cax=cax).ax.tick_params(labelsize=7)
        panelA_axes.append(ax)

    if panelA_axes:
        add_panel_label(panelA_axes[0], "A")
    mapping_parts = []
    for color in ("red", "green", "blue"):
        name = rgb_mapping.get(color)
        if name:
            mapping_parts.append(f"{long_aliases.get(name, name)} → {color}")
    add_caption(panel_a["x"], panel_a["y"], panel_a["w"], "A – Raw field of view (ND2)")
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
    panel_b = dict(x=panel_a["x"] + panel_a["w"] + ROW_GAP, y=row1_y, w=1244, h=ROW1_H)
    panel_b_inner_h = panel_b["h"] - CAPTION
    panel_b_inner_y = panel_b["y"] + CAPTION
    slot_w_b = (panel_b["w"] - 2 * sub_gap) / 3
    slot_h_b = (panel_b_inner_h - sub_gap) / 2

    sigma = seg_settings.get("smoothing_sigma", 1.2)
    otsu_offset = seg_settings.get("otsu_offset", 0.0)
    min_size_setting = seg_settings.get("min_size")
    max_size_setting = seg_settings.get("max_size")
    footprint_setting = seg_settings.get("peak_footprint")
    footprint_radius = None
    if footprint_setting:
        footprint_radius = (footprint_setting - 1) // 2

    raw_subtitle = "Nuclear stain"
    if seg.tophat_radius:
        raw_subtitle = f"White tophat r={seg.tophat_radius}px"

    clean_subtitle = "Remove noise, fill holes"
    if min_size_setting:
        clean_subtitle = f"Min area ≥ {int(min_size_setting)} px"
        if max_size_setting:
            clean_subtitle += f", max ≤ {int(max_size_setting)} px"
    elif max_size_setting:
        clean_subtitle = f"Max area ≤ {int(max_size_setting)} px"

    seeds_subtitle = "Find watershed peaks"
    if footprint_radius is not None:
        seeds_subtitle = f"Footprint r={footprint_radius}px"

    final_subtitle = "Final segmentation"
    if max_size_setting:
        final_subtitle += f" (max {int(max_size_setting)} px)"

    b_images = [
        (seg.raw, "B1. Raw DAPI", raw_subtitle, "gray", (0, 1)),
        (seg.smoothed, "B2. Gaussian smoothed", f"σ = {sigma:g}", "gray", (0, 1)),
        (seg.binary.astype(float), "B3. Otsu threshold", f"Offset Δ={otsu_offset:+.2f}", "gray", (0, 1)),
        (seg.cleaned.astype(float), "B4. Mask cleaned", clean_subtitle, "gray", (0, 1)),
        (_normalise_for_display(seg.distance), "B5. Distance + seeds", seeds_subtitle, "magma", None),
        (seg.labels, "B6. Labeled nuclei", final_subtitle, "turbo", None),
    ]

    panelB_axes: List[plt.Axes] = []
    for idx, (img, title, subtitle, cmap, norm_bounds) in enumerate(b_images):
        row = idx // 3
        col = idx % 3
        slot_x = panel_b["x"] + col * (slot_w_b + sub_gap)
        slot_y = panel_b_inner_y + (1 - row) * (slot_h_b + sub_gap)
        ax = add_axes_pixels(slot_x, slot_y, slot_w_b - (cbar_w + cbar_pad), slot_h_b)
        if title.startswith("B5"):
            ax.imshow(img, cmap=cmap)
            ax.contour(seg.markers > 0, colors="cyan", linewidths=0.6)
        elif title.startswith("B6"):
            ax.imshow(seg.raw, cmap="gray", alpha=0.6)
            masked = np.ma.masked_where(seg.labels == 0, seg.labels)
            ax.imshow(masked, cmap="turbo", alpha=0.45)
        else:
            ax.imshow(img, cmap=cmap)
        ax.axis("off")
        _add_scale_bar(ax, pixel_size_um, seg.raw.shape)
        ax.set_title(title, fontsize=9, pad=4)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes, fontsize=8, ha="center", va="top")

        cax = add_axes_pixels(slot_x + slot_w_b - cbar_w, slot_y, cbar_w, slot_h_b)
        if norm_bounds is None:
            norm = None
        else:
            norm = mcolors.Normalize(vmin=norm_bounds[0], vmax=norm_bounds[1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cax).ax.tick_params(labelsize=7)
        panelB_axes.append(ax)

    if panelB_axes:
        add_panel_label(panelB_axes[0], "B")
    add_caption(panel_b["x"], panel_b["y"], panel_b["w"], "B – Stepwise segmentation pipeline")

    # Panel C ------------------------------------------------------------
    panel_c = dict(x=MARGIN_LEFT, y=row2_y, w=780, h=ROW2_H)
    axC = add_axes_pixels(panel_c["x"], panel_c["y"] + CAPTION, panel_c["w"], panel_c["h"] - CAPTION)
    axC.imshow(patch_image, cmap="gray")
    axC.contour(patch_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    if patch_ring.any():
        axC.contour(patch_ring.astype(float), levels=[0.5], colors="orange", linestyles="--", linewidths=1.0)
    axC.axis("off")
    _add_scale_bar(axC, pixel_size_um, patch_image.shape)
    add_panel_label(axC, "C")
    measurement_channels = [long_aliases[name] for name in panel_channels if name != dapi_key and name in long_aliases]
    if measurement_channels:
        if len(measurement_channels) == 1:
            channel_text = measurement_channels[0]
        elif len(measurement_channels) == 2:
            channel_text = " & ".join(measurement_channels)
        else:
            channel_text = ", ".join(measurement_channels[:2]) + "…"
    else:
        channel_text = "additional markers"
    add_caption(panel_c["x"], panel_c["y"], panel_c["w"], f"C – Per-cell measurement region ({channel_text})")

    # Panel E ------------------------------------------------------------
    panel_e = dict(x=MARGIN_LEFT, y=row3_y, w=592, h=ROW3_H)
    axE = add_axes_pixels(panel_e["x"], panel_e["y"] + CAPTION, panel_e["w"], panel_e["h"] - CAPTION)
    axE.tick_params(labelsize=12)
    channel_cols = [col for col in cell_table.columns if col.startswith("mean_")]
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
        axE.set_xlabel("Mean intensity (a.u.)", fontsize=12)
        axE.set_ylabel("Density / cumulative", fontsize=12)
        axE.legend(fontsize=10)
    else:
        axE.text(0.5, 0.5, "No intensity data", ha="center", va="center", fontsize=12)
        axE.set_axis_off()
    add_panel_label(axE, "E")
    add_caption(panel_e["x"], panel_e["y"], panel_e["w"], "E – Marker distributions across cells")

    # Panel F ------------------------------------------------------------
    panel_f = dict(x=panel_e["x"] + panel_e["w"] + ROW_GAP, y=row3_y, w=592, h=ROW3_H)
    axF_width = panel_f["w"] - (cbar_w + cbar_pad)
    axF = add_axes_pixels(panel_f["x"], panel_f["y"] + CAPTION, axF_width, panel_f["h"] - CAPTION)
    axF.tick_params(labelsize=12)
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
        axF.set_xlabel(f"Mean {column_label_map.get(xcol, xcol.replace('mean_', ''))}", fontsize=12)
        axF.set_ylabel(f"Mean {column_label_map.get(ycol, ycol.replace('mean_', ''))}", fontsize=12)
        caxF = add_axes_pixels(panel_f["x"] + axF_width + cbar_pad, panel_f["y"] + CAPTION, cbar_w, panel_f["h"] - CAPTION)
        cb = fig.colorbar(scatter, cax=caxF)
        cb.ax.set_ylabel("Cell area (px)", fontsize=11)
        cb.ax.tick_params(labelsize=10)
    else:
        axF.text(0.5, 0.5, "Need ≥2 channels", ha="center", va="center", fontsize=12)
        axF.set_axis_off()
    add_panel_label(axF, "F")
    add_caption(panel_f["x"], panel_f["y"], panel_f["w"], "F – Marker co-expression per cell")

    # Panel G ------------------------------------------------------------
    panel_g = dict(x=panel_f["x"] + panel_f["w"] + ROW_GAP, y=row3_y, w=592, h=ROW3_H)
    axG = add_axes_pixels(panel_g["x"], panel_g["y"] + CAPTION, panel_g["w"], panel_g["h"] - CAPTION)
    axG.axis("off")
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
    axG.text(0.0, 1.0, outputs_text, family="monospace", fontsize=11, ha="left", va="top")
    add_panel_label(axG, "G")
    add_caption(panel_g["x"], panel_g["y"], panel_g["w"], "G – Workflow outputs")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
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
