"""Batch runner for Moran's I spatial autocorrelation across ROI snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from .project_io import load_roi_images
from .spatial_stats import (
    local_moran_map,
    local_moran_permutation_pvals,
    morans_i_snapshot,
    permutation_test_global,
)
from .utils import apply_roi_mask

CONFIG_PATH = Path(__file__).resolve().parent.parent / "dcxspot_config.json"


def _load_config() -> Dict[str, object]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _default_project_root() -> Optional[Path]:
    cfg = _load_config()
    project_root = cfg.get("project_root")
    if project_root:
        return Path(project_root).expanduser().resolve()
    return None


def _default_roi_measurements_path(project_root: Path) -> Path:
    path = project_root / "roi_measurements.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find roi_measurements.csv under {project_root}")
    return path


def _save_heatmap(data: np.ndarray, mask: np.ndarray, path: Path, title: str) -> None:
    cmap = colormaps.get_cmap("magma").with_extremes(bad="black")
    disp = np.full_like(data, np.nan, dtype=np.float32)
    disp[mask] = data[mask]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(disp, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _normalize_for_display(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = image[mask]
    norm = np.zeros_like(image, dtype=np.float32)
    if vals.size == 0:
        return norm
    lo, hi = np.percentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return norm
    norm_vals = np.clip((vals - lo) / (hi - lo), 0, 1)
    norm[mask] = norm_vals
    return norm


def _save_overlay(fluor_image: np.ndarray, mask: np.ndarray, local_map: np.ndarray, path: Path, alpha: float = 0.5) -> None:
    base_norm = _normalize_for_display(fluor_image, mask)
    base_rgb = np.repeat(base_norm[..., None], 3, axis=-1)

    local_vals = local_map[mask]
    if local_vals.size > 0:
        lo, hi = np.percentile(local_vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((local_map - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros_like(local_map)

    cmap = colormaps.get_cmap("viridis")
    overlay_rgb = cmap(norm)[..., :3]
    blended = base_rgb.copy()
    blended[mask] = (1 - alpha) * base_rgb[mask] + alpha * overlay_rgb[mask]
    blended = np.clip(blended, 0, 1)

    plt.imsave(path, blended)


def process_row(
    row: pd.Series,
    project_root: Path,
    neighbors: int,
    permutations: int,
    local_permutations: int,
    overlay_dir: Path,
    heatmap_dir: Path,
    pmap_dir: Path,
    random_state: Optional[int],
) -> Dict[str, object]:
    brightfield, mask, fluor = load_roi_images(row, project_root)
    rel = row.get("image_relpath")
    base_name = (
        Path(rel).stem
        if isinstance(rel, str) and rel
        else Path(row.get("image_path", "roi")).stem
    )
    fluor_masked = apply_roi_mask(fluor, mask, outside="zero")

    global_stats = permutation_test_global(
        fluor_masked,
        mask,
        neighbors=neighbors,
        permutations=permutations,
        random_state=random_state,
    )

    local_results: Dict[str, Path] = {}
    if heatmap_dir or overlay_dir or local_permutations > 0:
        I_obs, _, _, _, _, xc, neighbor_sum_xc, _ = morans_i_snapshot(
            fluor_masked,
            mask,
            neighbors=neighbors,
        )
        local_map = local_moran_map(xc, mask, neighbor_sum_xc)

        heatmap_path = heatmap_dir / f"{base_name}_local_moran.png"
        _save_heatmap(local_map, mask, heatmap_path, "Local Moran (Ii)")
        local_results["local_heatmap"] = heatmap_path

        overlay_path = overlay_dir / f"{base_name}_local_overlay.png"
        _save_overlay(fluor_masked, mask, local_map, overlay_path)
        local_results["local_overlay"] = overlay_path

        if local_permutations > 0:
            Ii_map, p_map = local_moran_permutation_pvals(
                fluor_masked,
                mask,
                neighbors=neighbors,
                local_permutations=local_permutations,
                random_state=random_state,
            )
            p_path = pmap_dir / f"{base_name}_local_pvals.png"
            _save_heatmap(p_map, mask, p_path, "Local Moran p-values")
            local_results["local_pmap"] = p_path

    record = {
        "image_relpath": row.get("image_relpath"),
        "well": row.get("well"),
        "day": row.get("day"),
        "time": row.get("time"),
    }
    record.update(global_stats)
    record.update(local_results)
    return record


def parse_args() -> argparse.Namespace:
    cfg_root = _default_project_root()

    parser = argparse.ArgumentParser(description="Batch Moran's I analysis for ROI fluorescence snapshots.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=cfg_root,
        help="Project root containing roi_measurements.csv (defaults to dcxspot_config.json).",
    )
    parser.add_argument(
        "--roi-measurements",
        type=Path,
        default=None,
        help="Explicit path to roi_measurements.csv (optional).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write the aggregated Moran results CSV (default: <project-root>/plots/morans/morans_global.csv).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        choices=[4, 8],
        default=8,
        help="Adjacency system for Moran's I (default: 8).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Global Moran permutation count (default: 999).",
    )
    parser.add_argument(
        "--local-permutations",
        type=int,
        default=199,
        help="Per-pixel permutation count for Local Moran p-values (default: 199).",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=Path,
        default=None,
        help="Directory to save local Moran heatmaps (default: <project-root>/plots/morans/heatmaps).",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Directory to save brightfield overlays (default: <project-root>/plots/morans/overlays).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows processed (for quick tests).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for permutations (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root
    if project_root is None:
        raise SystemExit("Project root not specified and not found in config." )
    project_root = project_root.expanduser().resolve()

    roi_path = args.roi_measurements or _default_roi_measurements_path(project_root)
    roi_path = roi_path.expanduser().resolve()

    output_csv = args.output_csv
    default_output_dir = project_root / "plots" / "morans"
    if output_csv is None:
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = default_output_dir / "morans_global.csv"
    else:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    heatmap_dir = (args.heatmap_dir or (default_output_dir / "heatmaps")).expanduser().resolve()
    overlay_dir = (args.overlay_dir or (default_output_dir / "overlays")).expanduser().resolve()
    pmap_dir = heatmap_dir / "p_values"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pmap_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(roi_path)
    if df.empty:
        raise SystemExit("roi_measurements.csv is empty.")

    rows = []
    wells = list(df.groupby('well', sort=True))
    total_wells = len(wells)
    processed_rows = 0
    row_limit = args.limit if args.limit is not None else len(df)

    for w_idx, (well, group) in enumerate(wells, start=1):
        if processed_rows >= row_limit:
            break

        group = group.sort_values(['day', 'time']) if {'day', 'time'} <= set(group.columns) else group
        print(f"\nProcessing well {well} ({w_idx}/{total_wells})")
        group_total = len(group)

        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            if processed_rows >= row_limit:
                break
            time_label = row.get('time', '')
            bar_width = 28
            progress = idx / group_total if group_total else 1.0
            filled = int(bar_width * progress)
            bar = '#' * filled + '-' * (bar_width - filled)
            print(f"  [{bar}] {idx}/{group_total} time {time_label}", end='\r', flush=True)

            rows.append(
                process_row(
                    row,
                    project_root=project_root,
                    neighbors=args.neighbors,
                    permutations=args.permutations,
                    local_permutations=args.local_permutations,
                    overlay_dir=overlay_dir,
                    heatmap_dir=heatmap_dir,
                    pmap_dir=pmap_dir,
                    random_state=args.random_state,
                )
            )
            processed_rows += 1

        print()  # newline after finishing the well

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Wrote Moran results for {len(result_df)} frames to {output_csv}")


if __name__ == "__main__":
    main()
