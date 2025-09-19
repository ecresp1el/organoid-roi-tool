"""Plot per-marker mean intensities across timepoints for a project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot absolute channel intensities grouped by timepoint",
    )
    parser.add_argument("project_root", type=Path, help="Project directory containing ND2 files")
    parser.add_argument(
        "--timepoint",
        dest="timepoints",
        action="append",
        default=None,
        help="Timepoint identifier substring (repeat flag for multiples). Default: DIV 18, DIV 23",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="TOKEN=LABEL",
        help="Override channel alias, e.g. --alias cy5=SOX2",
    )
    parser.add_argument(
        "--skip-token",
        action="append",
        default=["dapi"],
        help="Channel substring to skip (default: dapi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: <project_root>/analysis)",
    )
    return parser.parse_args(argv)


def _resolve_alias_overrides(items: Iterable[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            print(f"Warning: ignoring alias '{item}' (expected TOKEN=LABEL)", file=sys.stderr)
            continue
        token, label = item.split("=", 1)
        overrides[token.strip().lower()] = label.strip()
    return overrides


def _compute_channel_means(nd2_path: Path, load_projection) -> Dict[str, float]:
    projection, channel_names, _ = load_projection(nd2_path)
    means: Dict[str, float] = {}
    for idx, name in enumerate(channel_names):
        means[name] = float(np.mean(projection[idx]))
    return means


def _plot_marker(marker: str, timepoints: List[str], data: Mapping[str, List[float]], output_path: Path) -> None:
    available = [tp for tp in timepoints if data.get(tp)]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    values = [data[tp] for tp in available]
    positions = range(1, len(available) + 1)

    ax.boxplot(values, positions=positions, widths=0.4, patch_artist=True,
               boxprops=dict(facecolor="#ccccff", alpha=0.6),
               medianprops=dict(color="black"))

    for pos, vals in zip(positions, values):
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
        ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals, color="#3366cc", alpha=0.7, s=25)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(available, rotation=20)
    ax.set_ylabel("Mean intensity (a.u.)")
    ax.set_title(f"{marker} mean intensity by timepoint")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    _setup_imports()

    from cellcount_tools.nd2_manifest import discover_nd2_files
    from cellcount_tools.workflow_figure import (
        _load_nd2_projection,
        build_alias_table,
        _channel_alias,
    )
    from cellcount_tools.project_config import load_channel_aliases

    args = parse_args(argv or sys.argv[1:])
    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        print(f"Error: project root {project_root} not found", file=sys.stderr)
        return 2

    nd2_files = discover_nd2_files(project_root)
    if not nd2_files:
        print("No ND2 files discovered", file=sys.stderr)
        return 1

    project_aliases = load_channel_aliases(project_root)
    override_aliases = _resolve_alias_overrides(args.alias)
    alias_table = build_alias_table({**project_aliases, **override_aliases})

    if args.timepoints:
        timepoints_config = list(dict.fromkeys(args.timepoints))
    else:
        timepoints_config = ["DIV 18", "DIV 23"]
    timepoints_lower = [tp.lower() for tp in timepoints_config]
    skip_tokens = [tok.lower() for tok in (args.skip_token or [])]

    marker_values: Dict[str, Dict[str, List[float]]] = {}

    for nd2_path in nd2_files:
        channel_means = _compute_channel_means(nd2_path, _load_nd2_projection)

        stem_lower = nd2_path.stem.lower()
        timepoint_label = None
        for token_lower, token in zip(timepoints_lower, timepoints_config):
            if token_lower in stem_lower:
                timepoint_label = token
                break
        if timepoint_label is None:
            continue

        for channel_name, mean_val in channel_means.items():
            channel_lower = channel_name.lower()
            if any(skip in channel_lower for skip in skip_tokens):
                continue
            marker_short, marker_long = _channel_alias(channel_name, alias_table)
            marker_label = marker_long or marker_short
            marker_entry = marker_values.setdefault(marker_label, {tp: [] for tp in timepoints_config})
            marker_entry.setdefault(timepoint_label, []).append(mean_val)

    if not marker_values:
        print("No marker data collected", file=sys.stderr)
        return 1

    output_dir = (args.output_dir or (project_root / "analysis")).expanduser().resolve()

    for marker_label, timepoint_data in marker_values.items():
        output_path = output_dir / f"{marker_label.replace(' ', '_')}_intensity.png"
        _plot_marker(marker_label, timepoints_config, timepoint_data, output_path)
        print(f"Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
