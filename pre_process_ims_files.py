"""
Entry-point script for preprocessing Imaris ``.ims`` files in a project folder.

The script reads stitched ``.ims`` volumes, extracts channel metadata, generates
max-intensity projections (grayscale and colourized), and writes a consolidated
CSV summary. Outputs live alongside the source data under a dedicated
``pre_processed_files`` folder so that the raw project data remains untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from imaris_tools import export_directory, plot_folder_projection_grid

PROJECT_DEFAULT = Path("/Users/ecrespo/Desktop/nestin_dcx_pcdh19_kovswt")
OUTPUT_FOLDER_NAME = "pre_processed_files"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Imaris stitched volumes.")
    parser.add_argument(
        "--project-folder",
        type=Path,
        default=PROJECT_DEFAULT,
        help=(
            "Path to the project directory containing raw .ims files. "
            "Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--output-folder-name",
        default=OUTPUT_FOLDER_NAME,
        help=(
            "Name of the folder within the project directory where the preprocessed "
            "artifacts will be stored. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .ims files within the project folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing existing per-file export folders if they already exist. "
            "Only affects generated outputs; raw project data is never removed."
        ),
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip generation of the per-channel grid PDF.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip generation of per-image 1xN overview figures.",
    )
    parser.add_argument(
        "--grid-percentile",
        type=float,
        default=95.0,
        help="Percentile used to scale the colour mapping in the grid PDF (default: %(default)s).",
    )
    parser.add_argument(
        "--grid-dpi",
        type=int,
        default=150,
        help="DPI for the generated PDF when plotting the grid (default: %(default)s).",
    )
    parser.add_argument(
        "--overview-percentile",
        type=float,
        default=95.0,
        help="Percentile used to scale per-image overview figures (default: %(default)s).",
    )
    parser.add_argument(
        "--overview-dpi",
        type=int,
        default=150,
        help="DPI for per-image overview figures (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    project_folder: Path = args.project_folder.expanduser().resolve()

    if not project_folder.exists():
        print(f"[error] project folder {project_folder} does not exist.", file=sys.stderr)
        return 1
    if not project_folder.is_dir():
        print(f"[error] {project_folder} is not a directory.", file=sys.stderr)
        return 1

    output_root = (project_folder / args.output_folder_name).resolve()
    if project_folder == output_root:
        print("[error] output folder must differ from the project folder.", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Project folder: {project_folder}")
    print(f"[info] Output folder:  {output_root}")

    print("[step] Exporting per-channel projections and metadata...")
    try:
        csv_path = export_directory(
            project_folder,
            output_root=output_root,
            pattern="*.ims",
            recursive=args.recursive,
            overwrite=args.overwrite,
            verbose=True,
            save_overview=not args.skip_overview,
            overview_percentile=args.overview_percentile,
            overview_dpi=args.overview_dpi,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"[error] failed to export Imaris files: {exc}", file=sys.stderr)
        return 1

    print(f"[info] Metadata CSV written to: {csv_path}")

    if not args.skip_grid:
        grid_path = output_root / "channel_overview.pdf"
        try:
            print("[step] Generating channel overview grid...")
            plot_folder_projection_grid(
                project_folder,
                grid_path,
                pattern="*.ims",
                recursive=args.recursive,
                percentile=args.grid_percentile,
                dpi=args.grid_dpi,
            )
            print(f"[info] Grid PDF written to: {grid_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[error] failed to generate grid PDF: {exc}", file=sys.stderr)

    print("[done] Preprocessing completed successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
