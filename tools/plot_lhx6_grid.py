"""Create channel-wise grids for LHX6 projects with DIV18 normalised to DIV23 range."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LHX6 channel grids across DIV18 vs DIV23")
    parser.add_argument("project_root", type=Path, help="Project directory containing ND2 files")
    parser.add_argument(
        "--marker-token",
        type=str,
        default="gfp",
        help="Substring identifying the marker channel (default: gfp)",
    )
    parser.add_argument(
        "--div18-token",
        type=str,
        default="DIV 18",
        help="Substring used to detect DIV18 files (default: 'DIV 18')",
    )
    parser.add_argument(
        "--div23-token",
        type=str,
        default="DIV 23",
        help="Substring used to detect DIV23 files (default: 'DIV 23')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <project_root>/panel_a/lhx6_div18_vs_div23.png)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    _setup_imports()

    from cellcount_tools.workflow_figure import (
        _add_scale_bar,
        _channel_alias,
        _load_nd2_projection,
        _normalise_for_display,
        build_alias_table,
        PSEUDO_COLORMAPS,
    )
    from cellcount_tools.project_config import load_channel_aliases
    from cellcount_tools.nd2_manifest import discover_nd2_files

    args = parse_args(argv or sys.argv[1:])
    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        print(f"Error: project root {project_root} not found", file=sys.stderr)
        return 2

    nd2_files = discover_nd2_files(project_root)
    if not nd2_files:
        print("No ND2 files discovered", file=sys.stderr)
        return 1

    alias_table = build_alias_table(load_channel_aliases(project_root))

    marker_token = args.marker_token.lower()
    div18_token = args.div18_token.lower()
    div23_token = args.div23_token.lower()

    marker_vmin = np.inf
    marker_vmax = -np.inf
    file_infos: Dict[Path, Dict[str, object]] = {}

    # First pass: gather metadata and compute DIV23 marker range
    for nd2_path in nd2_files:
        projection, channel_names, pixel_size = _load_nd2_projection(nd2_path)
        info: Dict[str, object] = {
            "pixel_size": pixel_size,
            "channels": channel_names,
        }
        file_infos[nd2_path] = info

        marker_idx = next(
            (idx for idx, name in enumerate(channel_names) if marker_token in name.lower()),
            None,
        )
        info["marker_idx"] = marker_idx

        if marker_idx is not None and div23_token in nd2_path.stem.lower():
            data = projection[marker_idx]
            marker_vmin = min(marker_vmin, float(np.min(data)))
            marker_vmax = max(marker_vmax, float(np.max(data)))

    if marker_vmax <= marker_vmin:
        print("Could not determine marker intensity range from DIV23 files", file=sys.stderr)
        return 1

    # Second pass: collect normalised channel images for plotting
    div18_entries: List[Dict[str, object]] = []
    div23_entries: List[Dict[str, object]] = []
    dapi_label = _channel_alias("dapi", alias_table)[1]
    marker_label = None

    for nd2_path in nd2_files:
        info = file_infos[nd2_path]
        projection, channel_names, pixel_size = _load_nd2_projection(nd2_path)

        dapi_idx = next((idx for idx, name in enumerate(channel_names) if "dapi" in name.lower()), None)
        if dapi_idx is None:
            continue

        marker_idx = info.get("marker_idx")

        dapi_img = _normalise_for_display(projection[dapi_idx])

        marker_img = None
        if marker_idx is not None:
            channel_name = channel_names[marker_idx]
            if div18_token in nd2_path.stem.lower():
                denom = max(marker_vmax - marker_vmin, 1e-9)
                marker_img = np.clip((projection[marker_idx] - marker_vmin) / denom, 0, 1)
            else:
                marker_img = _normalise_for_display(projection[marker_idx])
            marker_label = _channel_alias(channel_name, alias_table)[1]
            marker_cmap = PSEUDO_COLORMAPS.get(marker_token, None)
        else:
            marker_cmap = None

        entry = {
            "file": nd2_path,
            "dapi": dapi_img,
            "marker": marker_img,
            "pixel_size": pixel_size,
            "marker_channel": marker_idx is not None,
            "marker_cmap": marker_cmap,
        }

        if div18_token in nd2_path.stem.lower():
            div18_entries.append(entry)
        elif div23_token in nd2_path.stem.lower():
            div23_entries.append(entry)

    if not div18_entries and not div23_entries:
        print("No matching DIV18 or DIV23 entries found", file=sys.stderr)
        return 1

    channels = [dapi_label]
    if marker_label:
        channels.append(marker_label)

    ages = [(args.div18_token, div18_entries), (args.div23_token, div23_entries)]
    ages = [age for age in ages if age[1]]

    max_cols = max(len(entries) for _, entries in ages)
    total_rows = len(channels) * len(ages)
    fig, axes = plt.subplots(total_rows, max_cols, figsize=(4.5 * max_cols, 4.5 * total_rows))
    if total_rows == 1:
        axes = np.array([axes])

    for age_idx, (age_label, entries) in enumerate(ages):
        for ch_idx, channel_name in enumerate(channels):
            row_index = age_idx * len(channels) + ch_idx
            row_axes = axes[row_index]
            if max_cols == 1:
                row_axes = [row_axes]
            for col_idx in range(max_cols):
                ax = row_axes[col_idx]
                if col_idx >= len(entries):
                    ax.axis("off")
                    continue
                entry = entries[col_idx]
                if channel_name == dapi_label:
                    img = entry["dapi"]
                    cmap = "gray"
                else:
                    img = entry["marker"]
                    if img is None:
                        ax.axis("off")
                        continue
                    cmap = entry.get("marker_cmap")
                    if cmap is None:
                        cmap = "viridis"

                ax.imshow(img, cmap=cmap)
                _add_scale_bar(ax, entry["pixel_size"], img.shape)
                if ch_idx == 0:
                    ax.set_title(entry["file"].stem, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"{age_label} – {channel_name}", fontsize=10)
                ax.axis("off")

    fig.tight_layout()
    output_path = args.output or (project_root / "panel_a" / "lhx6_div18_vs_div23.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Summary figure saved -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
