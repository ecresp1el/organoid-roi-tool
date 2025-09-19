"""Batch-generate Panel A figures for ND2 files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 1x4 Panel A figures for ND2 data")
    parser.add_argument("input", type=Path, help="ND2 file or directory containing ND2 files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of ND2 files to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for figures (default: <nd2_dir>/panel_a)",
    )
    return parser.parse_args(argv)


def _discover_nd2(paths_root: Path) -> list[Path]:
    from cellcount_tools.nd2_manifest import discover_nd2_files

    if paths_root.is_file():
        if paths_root.suffix.lower() != ".nd2":
            raise ValueError(f"Provided file is not an ND2: {paths_root}")
        return [paths_root]
    return discover_nd2_files(paths_root)


def main(argv: list[str] | None = None) -> int:
    _setup_imports()
    from cellcount_tools.panel_a_figure import generate_panel_a_figure

    args = parse_args(argv or sys.argv[1:])
    root = args.input.expanduser().resolve()
    if not root.exists():
        print(f"Error: {root} not found", file=sys.stderr)
        return 2

    try:
        nd2_files = _discover_nd2(root)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not nd2_files:
        print("No ND2 files discovered", file=sys.stderr)
        return 1

    limit = args.limit if args.limit is None or args.limit > 0 else None
    total = len(nd2_files) if limit is None else min(limit, len(nd2_files))
    if total == 0:
        print("No ND2 files selected after applying limit", file=sys.stderr)
        return 1

    completed = 0
    attempt = 0
    for nd2_path in nd2_files:
        if limit is not None and attempt >= limit:
            break
        attempt += 1
        try:
            figure_path = generate_panel_a_figure(
                nd2_path,
                output_dir=args.output_dir,
            )
        except Exception as exc:  # pragma: no cover - runtime reporting only
            print(
                f"[{attempt}/{total}] Skipped {nd2_path.name}: {exc}",
                file=sys.stderr,
            )
            continue

        completed += 1
        print(f"[{attempt}/{total}] Saved Panel A figure -> {figure_path}")

    if completed == 0:
        print("No Panel A figures were generated", file=sys.stderr)
        return 1

    print(f"Finished: {completed} figure(s) saved (requested limit={limit or 'all'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
