"""Generate a Panel A-style figure (merged RGB + single channels) for ND2 data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from .workflow_figure import (
    _add_scale_bar,
    _channel_alias,
    build_alias_table,
    _load_nd2_projection,
    _normalise_for_display,
    _panel_channel_sequence,
    _pseudo_colormap_for_channel,
    _build_rgb_image,
)
from .project_config import load_channel_aliases


def _get_channel_images(projection: Tuple[float], channel_names: Iterable[str]) -> Dict[str, float]:
    return {
        name: _normalise_for_display(projection[idx]) for idx, name in enumerate(channel_names)
    }


def generate_panel_a_figure(
    nd2_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    figure_name: str | None = None,
    channel_aliases: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Render a 1×4 panel figure (merged RGB, DAPI, Cy5, GFP) and save to disk."""

    nd2_path = Path(nd2_path).expanduser().resolve()
    if project_root is None:
        inferred_root = nd2_path.parent
    else:
        inferred_root = Path(project_root)
    project_root_path = inferred_root.expanduser().resolve()

    projection, channel_names, pixel_size_um = _load_nd2_projection(nd2_path)
    channel_images = _get_channel_images(projection, channel_names)
    rgb, rgb_mapping = _build_rgb_image(channel_images)
    config_aliases = load_channel_aliases(project_root_path)
    overrides = dict(config_aliases)
    if channel_aliases:
        overrides.update({k.lower(): v for k, v in channel_aliases.items()})
    alias_table = build_alias_table(overrides)
    panel_channels = _panel_channel_sequence(channel_names)

    dapi_key = next((n for n in panel_channels if "dapi" in n.lower()), panel_channels[-1])
    cy5_key = rgb_mapping.get("red", dapi_key)
    gfp_key = rgb_mapping.get("green", dapi_key)

    def resolved_channel(key: str, fallback: str) -> str:
        return key if key in channel_images else fallback

    dapi_key = resolved_channel(dapi_key, panel_channels[0])
    cy5_key = resolved_channel(cy5_key, dapi_key)
    gfp_key = resolved_channel(gfp_key, dapi_key)

    channel_labels = {name: _channel_alias(name, alias_table)[1] for name in channel_names}

    def label_for(key: str) -> str:
        return channel_labels.get(key, _channel_alias(key, alias_table)[1])

    single_channels: List[Tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for name in panel_channels:
        if name == dapi_key or name not in channel_images or name in seen:
            continue
        seen.add(name)
        single_channels.append((name, channel_images[name]))

    panel_defs: List[Tuple[np.ndarray, str, object]] = [
        (rgb, "Merged RGB", None),
        (channel_images[dapi_key], label_for(dapi_key), _pseudo_colormap_for_channel(dapi_key) or "gray"),
    ]

    priority_tokens = ["cy5", "sox2", "dcx", "gfp", "lhx6"]

    def channel_priority(name: str) -> int:
        lname = name.lower()
        for idx, token in enumerate(priority_tokens):
            if token in lname:
                return idx
        return len(priority_tokens)

    single_channels_sorted = sorted(single_channels, key=lambda item: channel_priority(item[0]))

    for name, img in single_channels_sorted:
        if name == dapi_key:
            continue
        panel_defs.append(
            (img, label_for(name), _pseudo_colormap_for_channel(name) or "gray")
        )

    if len(panel_defs) == 2:  # DAPI only
        pass

    if output_dir is None:
        output_dir = project_root_path / "panel_a"

    if output_dir is None:
        output_dir = nd2_path.parent / "panel_a"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if figure_name is None:
        figure_name = f"{nd2_path.stem}_panelA.png"
    figure_path = output_dir / figure_name

    total_panels = len(panel_defs)
    fig, axes = plt.subplots(1, total_panels, figsize=(4.2 * total_panels, 4.5))
    if total_panels == 1:
        axes = [axes]
    for ax, (img, title, cmap) in zip(axes, panel_defs):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if cmap is not None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
        _add_scale_bar(ax, pixel_size_um, img.shape)

    fig.suptitle(nd2_path.name, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)

    return figure_path
