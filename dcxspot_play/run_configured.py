"""One-tap runner that loads dcxspot_config.json and executes the pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .driver import run_project


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(
            f"Config file not found: {config_path}\n"
            "Create one based on dcxspot_config.json.sample or update the path."
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "dcxspot_config.json"
    cfg = load_config(config_path)

    project_root = Path(cfg["project_root"])
    output_root = Path(cfg["output_root"])

    summary = run_project(
        project_root=project_root,
        fluor_subdir=cfg.get("fluor_subdir", "fluorescence"),
        fluor_suffix=cfg.get("fluor_suffix", "_mcherry.tif"),
        output_root=output_root,
        limit=cfg.get("limit"),
        min_area=int(cfg.get("min_area", 24)),
        max_area=int(cfg.get("max_area", 8000)),
        min_distance=int(cfg.get("min_distance", 3)),
        morph_radius=int(cfg.get("morph_radius", 0)),
        normalize_percentiles=tuple(cfg.get("normalize_percentiles", (0.0, 0.0))),
        save_1x4=bool(cfg.get("save_1x4", False)),
        verbose=True,
    )

    print("\nFinished configured run")
    print(f"  inspected : {summary.inspected}")
    print(f"  processed : {summary.processed}")
    print(f"  skipped   : {summary.skipped}")
    print(f"  outputs   : {summary.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
