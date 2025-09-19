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


def _get_channel_images(
    projection: np.ndarray,
    channel_names: Iterable[str],
    scaling_overrides: Mapping[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(channel_names):
        arr = projection[idx]
        if scaling_overrides and name in scaling_overrides:
            vmin, vmax = scaling_overrides[name]
            denom = max(vmax - vmin, 1e-9)
            arr = np.clip((arr - vmin) / denom, 0.0, 1.0)
        else:
            arr = _normalise_for_display(arr)
        images[name] = arr
    return images


def generate_panel_a_figure(
    nd2_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    figure_name: str | None = None,
    channel_aliases: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    channel_scaling: Mapping[str, Tuple[float, float]] | None = None,
) -> Path:
    """Render a 1×4 panel figure (merged RGB, DAPI, Cy5, GFP) and save to disk."""

    nd2_path = Path(nd2_path).expanduser().resolve()
    if project_root is None:
        inferred_root = nd2_path.parent
    else:
        inferred_root = Path(project_root)
    project_root_path = inferred_root.expanduser().resolve()

    projection, channel_names, pixel_size_um = _load_nd2_projection(nd2_path)
    channel_images = _get_channel_images(projection, channel_names, channel_scaling)
    rgb, rgb_mapping = _build_rgb_image(channel_images)
    config_aliases = load_channel_aliases(project_root_path)
    overrides = dict(config_aliases)
    if channel_aliases:
        overrides.update({k.lower(): v for k, v in channel_aliases.items()})
    alias_table = build_alias_table(overrides)
    panel_channels = _panel_channel_sequence(channel_names)

    dapi_key = next((n for n in panel_channels if "dapi" in n.lower()), panel_channels[0])
    if dapi_key not in channel_images:
        dapi_key = panel_channels[0]

    channel_labels = {name: _channel_alias(name, alias_table)[1] for name in channel_names}

    def label_for(key: str) -> str:
        return channel_labels.get(key, _channel_alias(key, alias_table)[1])

    def channel_priority(name: str) -> int:
        lname = name.lower()
        tokens = ["cy5", "sox2", "dcx", "gfp", "lhx6", "ki67"]
        for idx, token in enumerate(tokens):
            if token in lname:
                return idx
        return len(tokens)

    non_dapi_channels = [name for name in channel_images.keys() if name != dapi_key]
    non_dapi_channels.sort(key=channel_priority)

    panel_defs: List[Tuple[np.ndarray, str, object]] = [
        (rgb, "Merged RGB", None),
        (channel_images[dapi_key], label_for(dapi_key), _pseudo_colormap_for_channel(dapi_key) or "gray"),
    ]

    for name in non_dapi_channels:
        img = channel_images[name]
        panel_defs.append((img, label_for(name), _pseudo_colormap_for_channel(name) or "gray"))

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
