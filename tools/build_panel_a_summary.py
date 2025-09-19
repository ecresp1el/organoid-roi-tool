"""Create grouped Panel A summary figures for a project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped Panel A figures")
    parser.add_argument("project_root", type=Path, help="Project directory containing ND2 files")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["DIV 13", "DIV 23"],
        help="Group identifiers searched in filenames (default: DIV 13 DIV 23)",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="TOKEN=LABEL",
        help="Override channel alias (e.g. --alias cy5=SOX2)",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="panel_a_summary.png",
        help="Filename for the combined summary figure",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _setup_imports()

    from cellcount_tools.nd2_manifest import discover_nd2_files
    from cellcount_tools.panel_a_figure import generate_panel_a_figure

    args = parse_args(argv or sys.argv[1:])
    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        print(f"Error: project root {project_root} not found", file=sys.stderr)
        return 2

    nd2_files = discover_nd2_files(project_root)
    if not nd2_files:
        print("No ND2 files discovered", file=sys.stderr)
        return 1

    alias_overrides: dict[str, str] = {}
    for item in args.alias:
        if "=" not in item:
            print(f"Warning: ignoring alias '{item}' (expected TOKEN=LABEL)", file=sys.stderr)
            continue
        token, label = item.split("=", 1)
        alias_overrides[token.strip().lower()] = label.strip()

    panel_dir = project_root / "panel_a"
    panel_dir.mkdir(parents=True, exist_ok=True)

    panel_images: dict[Path, Path] = {}
    for nd2_path in nd2_files:
        panel_path = generate_panel_a_figure(
            nd2_path,
            output_dir=panel_dir,
            channel_aliases=alias_overrides or None,
            project_root=project_root,
        )
        panel_images[nd2_path] = panel_path
        print(f"Saved Panel A figure -> {panel_path}")

    groups = args.groups or ["DIV"]
    grouped_panels: dict[str, list[Path]] = {group: [] for group in groups}
    remainder: list[Path] = []

    for nd2_path, panel_path in panel_images.items():
        stem_lower = nd2_path.stem.lower()
        matched = False
        for group in groups:
            if group.lower() in stem_lower:
                grouped_panels[group].append(panel_path)
                matched = True
                break
        if not matched:
            remainder.append(panel_path)

    rows = [(group, grouped_panels[group]) for group in groups if grouped_panels[group]]
    if remainder:
        rows.append(("Other", remainder))

    if not rows:
        print("No panel images matched the specified groups", file=sys.stderr)
        return 1

    max_cols = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(len(rows), max_cols, figsize=(4.5 * max_cols, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for row_idx, (group, images) in enumerate(rows):
        row_axes = axes[row_idx]
        if max_cols == 1:
            row_axes = [row_axes]
        for col_idx in range(max_cols):
            ax = row_axes[col_idx]
            if col_idx < len(images):
                img = mpimg.imread(images[col_idx])
                ax.imshow(img)
                ax.set_title(Path(images[col_idx]).stem, fontsize=10)
            else:
                ax.imshow([[0, 0], [0, 0]], cmap="gray")
                ax.set_visible(False)
            ax.axis("off")
        row_axes[0].set_ylabel(group, fontsize=12, rotation=90, labelpad=30)

    fig.tight_layout()
    summary_path = panel_dir / args.summary_name
    fig.savefig(summary_path, dpi=250)
    plt.close(fig)
    print(f"Summary figure saved -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

