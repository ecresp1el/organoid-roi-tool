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


def _save_overlay(brightfield: np.ndarray, mask: np.ndarray, local_map: np.ndarray, path: Path, alpha: float = 0.5) -> None:
    bf = brightfield.astype(np.float32, copy=False)
    bf_norm = bf - np.nanmin(bf)
    if bf_norm.max() > 0:
        bf_norm /= bf_norm.max()
    base_rgb = np.stack([bf_norm, bf_norm, bf_norm], axis=-1)

    local_vals = local_map[mask]
    if local_vals.size > 0:
        lo, hi = np.percentile(local_vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((local_map - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros_like(local_map)

    cmap = colormaps.get_cmap("viridis")
    overlay = cmap(norm)[..., :3]
    blended = base_rgb.copy()
    blended[mask] = (1 - alpha) * base_rgb[mask] + alpha * overlay[mask]
    blended = np.clip(blended, 0, 1)

    plt.imsave(path, blended)


def process_row(
    row: pd.Series,
    project_root: Path,
    neighbors: int,
    permutations: int,
    local_permutations: int,
    overlay_dir: Optional[Path],
    heatmap_dir: Optional[Path],
    random_state: Optional[int],
) -> Dict[str, object]:
    brightfield, mask, fluor = load_roi_images(row, project_root)
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

        if heatmap_dir:
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = heatmap_dir / f"{Path(row.image_relpath).stem}_local_moran.png"
            _save_heatmap(local_map, mask, heatmap_path, "Local Moran (Ii)")
            local_results["local_heatmap"] = heatmap_path

        if overlay_dir:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / f"{Path(row.image_relpath).stem}_local_overlay.png"
            _save_overlay(brightfield, mask, local_map, overlay_path)
            local_results["local_overlay"] = overlay_path

        if local_permutations > 0:
            p_heatmap_dir = heatmap_dir / "p_values" if heatmap_dir else None
            Ii_map, p_map = local_moran_permutation_pvals(
                fluor_masked,
                mask,
                neighbors=neighbors,
                local_permutations=local_permutations,
                random_state=random_state,
            )
            if p_heatmap_dir:
                p_heatmap_dir.mkdir(parents=True, exist_ok=True)
                p_path = p_heatmap_dir / f"{Path(row.image_relpath).stem}_local_pvals.png"
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
        default=0,
        help="If >0, run per-pixel permutation test for local Moran (costly).",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=Path,
        default=None,
        help="Directory to save local Moran heatmaps (optional).",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Directory to save brightfield overlays (optional).",
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

    heatmap_dir = args.heatmap_dir
    overlay_dir = args.overlay_dir
    if heatmap_dir:
        heatmap_dir = heatmap_dir.expanduser().resolve()
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir = overlay_dir.expanduser().resolve()
        overlay_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(roi_path)
    if df.empty:
        raise SystemExit("roi_measurements.csv is empty.")

    rows = []
    processed_count = 0
    for _, row in df.iterrows():
        rows.append(
            process_row(
                row,
                project_root=project_root,
                neighbors=args.neighbors,
                permutations=args.permutations,
                local_permutations=args.local_permutations,
                overlay_dir=overlay_dir,
                heatmap_dir=heatmap_dir,
                random_state=args.random_state,
            )
        )
        processed_count += 1
        if args.limit and processed_count >= args.limit:
            break

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Wrote Moran results for {len(result_df)} frames to {output_csv}")


if __name__ == "__main__":
    main()
