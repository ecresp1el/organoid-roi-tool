"""Generate DIV18 vs DIV23 grids for Nestin (GFP) and DCX (Cy5)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Nestin/DCX DIV18 vs DIV23 grids")
    parser.add_argument("project_root", type=Path, help="Project directory containing ND2 files")
    parser.add_argument(
        "--div18-token",
        type=str,
        default="DIV 18",
        help="Substring that identifies DIV18 files (default: 'DIV 18')",
    )
    parser.add_argument(
        "--div23-token",
        type=str,
        default="DIV 23",
        help="Substring that identifies DIV23 files (default: 'DIV 23')",
    )
    parser.add_argument(
        "--nestin-token",
        type=str,
        default="gfp",
        help="Substring identifying the Nestin channel (default: 'gfp')",
    )
    parser.add_argument(
        "--dcx-token",
        type=str,
        default="cy5",
        help="Substring identifying the DCX channel (default: 'cy5')",
    )
    parser.add_argument(
        "--output-nestin",
        type=Path,
        default=None,
        help="Output PNG for Nestin grid (default: <project>/panel_a/nestin_div18_vs_div23.png)",
    )
    parser.add_argument(
        "--output-dcx",
        type=Path,
        default=None,
        help="Output PNG for DCX grid (default: <project>/panel_a/dcx_div18_vs_div23.png)",
    )
    return parser.parse_args(argv)


def _normalise_marker(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-9)
    return np.clip((arr - vmin) / denom, 0.0, 1.0)


def _collect_entries(
    project_root: Path,
    nd2_files: Iterable[Path],
    marker_token: str,
    div18_token: str,
    div23_token: str,
    alias_table,
    load_projection,
    normalise_image,
    add_scale_bar,
    pseudo_colormaps,
) -> Tuple[List[dict], List[dict], str, str]:
    marker_token = marker_token.lower()
    div18_token = div18_token.lower()
    div23_token = div23_token.lower()

    div23_vmin = np.inf
    div23_vmax = -np.inf

    cached_projections: Dict[Path, Tuple[np.ndarray, List[str], float | None]] = {}

    for nd2_path in nd2_files:
        projection, channel_names, pixel_size = load_projection(nd2_path)
        cached_projections[nd2_path] = (projection, channel_names, pixel_size)
        for idx, name in enumerate(channel_names):
            if marker_token in name.lower() and div23_token in nd2_path.stem.lower():
                data = projection[idx]
                div23_vmin = min(div23_vmin, float(data.min()))
                div23_vmax = max(div23_vmax, float(data.max()))

    if not np.isfinite(div23_vmin) or div23_vmax <= div23_vmin:
        raise RuntimeError(f"No DIV23 data found for marker token '{marker_token}'")

    div18_entries: List[dict] = []
    div23_entries: List[dict] = []
    marker_label = None
    dapi_label = None
    marker_cmap = pseudo_colormaps.get(marker_token, None)

    from cellcount_tools.workflow_figure import _channel_alias

    for nd2_path, (projection, channel_names, pixel_size) in cached_projections.items():
        labels = [_channel_alias(name, alias_table)[1] for name in channel_names]
        dapi_idx = next((idx for idx, name in enumerate(channel_names) if "dapi" in name.lower()), None)
        if dapi_idx is None:
            continue
        dapi_label = labels[dapi_idx]
        marker_idx = next((idx for idx, name in enumerate(channel_names) if marker_token in name.lower()), None)
        if marker_idx is None:
            continue
        marker_label = labels[marker_idx]

        dapi_img = normalise_image(projection[dapi_idx])
        if div18_token in nd2_path.stem.lower():
            marker_img = _normalise_marker(projection[marker_idx], div23_vmin, div23_vmax)
            target = div18_entries
        elif div23_token in nd2_path.stem.lower():
            marker_img = normalise_image(projection[marker_idx])
            target = div23_entries
        else:
            continue

        target.append(
            {
                "file": nd2_path,
                "dapi": dapi_img,
                "marker": marker_img,
                "pixel_size": pixel_size,
                "marker_cmap": marker_cmap,
            }
        )

    return div18_entries, div23_entries, dapi_label or "DAPI", marker_label or marker_token.upper()


def _plot_grid(
    output_path: Path,
    div18_entries: List[dict],
    div23_entries: List[dict],
    dapi_label: str,
    marker_label: str,
    add_scale_bar,
):
    ages = [("DIV 18", div18_entries), ("DIV 23", div23_entries)]
    ages = [age for age in ages if age[1]]
    if not ages:
        raise RuntimeError("No entries to plot")

    channels = [dapi_label, marker_label]
    max_cols = max(len(entries) for _, entries in ages)
    total_rows = len(ages) * len(channels)

    fig, axes = plt.subplots(total_rows, max_cols, figsize=(4.5 * max_cols, 4.5 * total_rows))
    if total_rows == 1:
        axes = np.array([axes])

    row_idx = 0
    for age_label, entries in ages:
        for channel_idx, channel_name in enumerate(channels):
            row_axes = axes[row_idx]
            if max_cols == 1:
                row_axes = [row_axes]
            for col_idx in range(max_cols):
                ax = row_axes[col_idx]
                if col_idx >= len(entries):
                    ax.axis("off")
                    continue
                entry = entries[col_idx]
                if channel_idx == 0:
                    img = entry["dapi"]
                    cmap = "gray"
                else:
                    img = entry["marker"]
                    cmap = entry.get("marker_cmap") or "viridis"
                ax.imshow(img, cmap=cmap)
                add_scale_bar(ax, entry["pixel_size"], img.shape)
                if channel_idx == 0:
                    ax.set_title(entry["file"].stem, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"{age_label} – {channel_name}", fontsize=10)
                ax.axis("off")
            row_idx += 1

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main_wrapper(argv: List[str] | None = None) -> int:
    _setup_imports()

    from cellcount_tools.workflow_figure import (
        _load_nd2_projection,
        _normalise_for_display,
        _add_scale_bar,
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

    try:
        div18_nestin, div23_nestin, dapi_label, nestin_label = _collect_entries(
            project_root,
            nd2_files,
            args.nestin_token,
            args.div18_token,
            args.div23_token,
            alias_table,
            _load_nd2_projection,
            _normalise_for_display,
            _add_scale_bar,
            PSEUDO_COLORMAPS,
        )
    except RuntimeError as exc:
        print(f"Skipping Nestin plot: {exc}", file=sys.stderr)
        div18_nestin, div23_nestin = [], []
        nestin_label = args.nestin_token.upper()
        dapi_label = "DAPI"

    try:
        div18_dcx, div23_dcx, dapi_label_dcx, dcx_label = _collect_entries(
            project_root,
            nd2_files,
            args.dcx_token,
            args.div18_token,
            args.div23_token,
            alias_table,
            _load_nd2_projection,
            _normalise_for_display,
            _add_scale_bar,
            PSEUDO_COLORMAPS,
        )
    except RuntimeError as exc:
        print(f"Skipping DCX plot: {exc}", file=sys.stderr)
        div18_dcx, div23_dcx = [], []
        dcx_label = args.dcx_token.upper()
        dapi_label_dcx = "DAPI"

    panel_dir = project_root / "panel_a"
    panel_dir.mkdir(parents=True, exist_ok=True)

    if div18_nestin or div23_nestin:
        output_nestin = args.output_nestin or (panel_dir / "nestin_div18_vs_div23.png")
        _plot_grid(output_nestin, div18_nestin, div23_nestin, dapi_label, nestin_label, _add_scale_bar)
        print(f"Nestin summary -> {output_nestin}")

    if div18_dcx or div23_dcx:
        output_dcx = args.output_dcx or (panel_dir / "dcx_div18_vs_div23.png")
        _plot_grid(output_dcx, div18_dcx, div23_dcx, dapi_label_dcx, dcx_label, _add_scale_bar)
        print(f"DCX summary -> {output_dcx}")

    if not ((div18_nestin or div23_nestin) or (div18_dcx or div23_dcx)):
        print("No plots were generated", file=sys.stderr)
        return 1

    return 0


def main(argv: List[str] | None = None) -> int:
    return main_wrapper(argv)


if __name__ == "__main__":
    raise SystemExit(main())
