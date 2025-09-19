"""Command-line helper to render the ND2 workflow figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the multi-panel workflow figure for a given ND2 file",
    )
    parser.add_argument("nd2_path", type=Path, help="Path to the source ND2 file")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root directory for outputs (defaults to <nd2_dir>/cellcount_project)",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=None,
        help="Override the output figure filename",
    )
    parser.add_argument(
        "--figure-subdir",
        type=str,
        default="figures",
        help="Subdirectory under the project root for the figure (default: figures)",
    )
    parser.add_argument(
        "--cells-subdir",
        type=str,
        default="cells",
        help="Subdirectory under the project root for the per-cell CSV (default: cells)",
    )

    seg_group = parser.add_argument_group("Segmentation overrides")
    seg_group.add_argument(
        "--smoothing-sigma",
        type=float,
        default=None,
        help="Gaussian smoothing sigma applied to DAPI before thresholding",
    )
    seg_group.add_argument(
        "--otsu-offset",
        type=float,
        default=None,
        help="Offset added to the Otsu threshold (positive => higher threshold)",
    )
    seg_group.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Minimum connected-component size (pixels) kept as nuclei",
    )
    seg_group.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Minimum distance between peak_local_max seeds (pixels)",
    )
    seg_group.add_argument(
        "--peak-footprint",
        type=int,
        default=None,
        help="Square footprint size for peak_local_max (set 0/None for default)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _setup_imports()
    from cellcount_tools import generate_workflow_figure

    args = parse_args(argv or sys.argv[1:])

    nd2_path = args.nd2_path.expanduser().resolve()
    if not nd2_path.exists():
        print(f"Error: ND2 path not found -> {nd2_path}", file=sys.stderr)
        return 2

    seg_params = {}
    if args.smoothing_sigma is not None:
        seg_params["smoothing_sigma"] = args.smoothing_sigma
    if args.otsu_offset is not None:
        seg_params["otsu_offset"] = args.otsu_offset
    if args.min_size is not None:
        seg_params["min_size"] = args.min_size
    if args.min_distance is not None:
        seg_params["min_distance"] = args.min_distance
    if args.peak_footprint is not None:
        seg_params["peak_footprint"] = args.peak_footprint

    figure_path = generate_workflow_figure(
        nd2_path,
        project_root=args.project_root,
        figure_subdir=args.figure_subdir,
        cells_subdir=args.cells_subdir,
        figure_name=args.figure_name,
        segmentation_settings=seg_params or None,
    )

    print(f"Workflow figure written to: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

