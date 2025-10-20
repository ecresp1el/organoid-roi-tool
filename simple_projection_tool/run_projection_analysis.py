"""CLI helper for launching immunohistochemistry projection analyses.

The command performs three high-level tasks:

1. Validate that ``<base-path>/simple_projections`` exists. If it does not, the
   helper transparently launches :mod:`simple_channel_projections` so downstream
   analyses always have the required TIFF exports.
2. Instantiate the requested analysis class from
   :mod:`simple_projection_tool.ihc_analyses` and run it end-to-end.
3. Print a human-readable summary of every CSV and figure generated during the
   run so scientists can verify the outputs without digging through folders.

The helper therefore provides a single command for both generating projections
and executing domain-specific post-processing pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type

from ihc_analyses import ANALYSIS_REGISTRY, ProjectionAnalysis

DEFAULT_BASE_PATH = Path(
    "/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-projection analyses on simple projection outputs. "
            "If the required simple_projections folder is missing, the script "
            "automatically launches simple_channel_projections first."
        ),
    )
    parser.add_argument(
        "analysis",
        choices=sorted(ANALYSIS_REGISTRY.keys()),
        help="Name of the analysis object to execute.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help=(
            "Root directory containing the simple_projection outputs for the"
            f" analysis (default: {DEFAULT_BASE_PATH})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store manifests, processed data, and plots.",
    )
    parser.add_argument(
        "--channel",
        dest="channels",
        action="append",
        default=None,
        help=(
            "Restrict the analysis to specific channel name(s). "
            "Repeat the flag to include multiple channels. "
            "Defaults are analysis-specific."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_cls: Type[ProjectionAnalysis] = ANALYSIS_REGISTRY[args.analysis]
    analysis = analysis_cls(
        base_path=args.base_path,
        output_dir=args.output_dir,
        channel_filter=args.channels,
    )
    analysis.run()

    saved_tables = getattr(analysis, "saved_table_paths", [])
    if saved_tables:
        print("[outputs] Data tables:")
        for path in saved_tables:
            print(f"    - {path}")
    else:
        print("[outputs] No CSV tables were generated.")

    saved_figures = getattr(analysis, "saved_figure_paths", [])
    if saved_figures:
        print("[outputs] Figures:")
        for path in saved_figures:
            print(f"    - {path}")
    else:
        print("[outputs] No figures were generated.")

    pipeline_dir = getattr(analysis, "pipeline_dir", analysis.output_dir)
    print(
        f"[done] {analysis.name} analysis complete. Data and plots are under {pipeline_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
