"""Batch runner for Moran's I spatial autocorrelation across ROI snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, colors

from .project_io import load_detection_labels, load_roi_images, resolve_brightfield_path
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


def _roi_bbox(mask: np.ndarray, pad: int = 4) -> Tuple[int, int, int, int]:
    mask_bool = mask.astype(bool, copy=False)
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, ys.min() - pad)
    y1 = min(mask_bool.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask_bool.shape[1], xs.max() + pad + 1)
    return y0, y1, x0, x1


def _render_overlay_arrays(
    fluor_image: np.ndarray,
    mask: np.ndarray,
    local_map: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_norm = _normalize_for_display(fluor_image, mask)
    base_rgb = np.repeat(base_norm[..., None], 3, axis=-1)

    local_vals = local_map[mask]
    local_norm = np.zeros_like(local_map, dtype=np.float32)
    if local_vals.size > 0:
        lo, hi = np.percentile(local_vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        local_norm_vals = np.clip((local_vals - lo) / (hi - lo), 0, 1)
        local_norm[mask] = local_norm_vals

    cmap = colormaps.get_cmap("viridis")
    local_rgb = cmap(local_norm)[..., :3]

    blended = base_rgb.copy()
    blended[mask] = (1 - alpha) * base_rgb[mask] + alpha * local_rgb[mask]
    blended = np.clip(blended, 0, 1)
    return base_norm, local_norm, local_rgb, blended




def _parse_day(day_label: str) -> Optional[int]:
    if not isinstance(day_label, str):
        return None
    try:
        prefix, value = day_label.split("_", 1)
        if prefix.lower() != "day":
            return None
        return int(value)
    except ValueError:
        return None


def _parse_time_to_hours(time_label: str) -> Optional[float]:
    if not isinstance(time_label, str):
        return None
    if "h" not in time_label or "m" not in time_label:
        return None
    try:
        hours_part, minutes_part = time_label.split("h", 1)
        minutes_part = minutes_part.rstrip("m")
        hours = int(hours_part)
        minutes = int(minutes_part) if minutes_part else 0
        return hours + minutes / 60.0
    except ValueError:
        return None


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
) -> Tuple[Dict[str, object], Dict[str, object]]:
    brightfield_path = resolve_brightfield_path(row, project_root)
    brightfield, mask, fluor = load_roi_images(row, project_root)
    mask = mask.astype(bool, copy=False)
    base_name = brightfield_path.stem
    fluor_masked = apply_roi_mask(fluor, mask, outside="zero")

    global_stats = permutation_test_global(
        fluor_masked,
        mask,
        neighbors=neighbors,
        permutations=permutations,
        random_state=random_state,
    )

    panel_entry: Dict[str, object] = {
        "time_label": row.get("time", ""),
        "time_hours": row.get("time_hours"),
    }
    local_results: Dict[str, Path] = {}

    detection_labels = load_detection_labels(brightfield_path)
    detection_mask = (
        detection_labels.astype(bool)
        if detection_labels is not None and detection_labels.shape == mask.shape
        else np.zeros_like(mask, dtype=bool)
    )

    base_norm = _normalize_for_display(fluor_masked, mask)
    base_norm = np.where(mask, base_norm, 0.0)

    if heatmap_dir or overlay_dir or local_permutations > 0:
        _, _, _, _, _, xc, neighbor_sum_xc, _ = morans_i_snapshot(
            fluor_masked, mask, neighbors=neighbors
        )
        local_map = local_moran_map(xc, mask, neighbor_sum_xc)

        heatmap_path = heatmap_dir / f"{base_name}_local_moran.png"
        _save_heatmap(local_map, mask, heatmap_path, "Local Moran (Ii)")
        local_results["local_heatmap"] = heatmap_path

        _, local_norm, _, overlay_rgb = _render_overlay_arrays(
            fluor_masked, mask, local_map
        )

        overlay_path = overlay_dir / f"{base_name}_local_overlay.png"
        plt.imsave(overlay_path, overlay_rgb)
        local_results["local_overlay"] = overlay_path

        p_map = None
        if local_permutations > 0:
            Ii_map, p_vals = local_moran_permutation_pvals(
                fluor_masked,
                mask,
                neighbors=neighbors,
                local_permutations=local_permutations,
                random_state=random_state,
            )
            local_map = Ii_map
            p_path = pmap_dir / f"{base_name}_local_pvals.png"
            _save_heatmap(p_vals, mask, p_path, "Local Moran p-values")
            local_results["local_pmap"] = p_path
            p_map = p_vals

        y0, y1, x0, x1 = _roi_bbox(mask)
        mask_crop = mask[y0:y1, x0:x1]
        raw_display = base_norm[y0:y1, x0:x1].copy()
        raw_display[~mask_crop] = 0.0

        local_map = np.where(mask, local_map, np.nan)
        local_raw_crop = local_map[y0:y1, x0:x1]
        local_norm_crop = local_norm[y0:y1, x0:x1]
        local_norm_crop = np.where(mask_crop, local_norm_crop, np.nan)

        detection_crop = (detection_mask & mask)[y0:y1, x0:x1]

        p_crop = None
        if p_map is not None:
            p_map = np.where(mask, p_map, np.nan)
            p_crop = p_map[y0:y1, x0:x1]
            p_crop = np.where(mask_crop, p_crop, np.nan)

        panel_entry.update(
            {
                "raw_display": raw_display,
                "mask_crop": mask_crop,
                "island_mask": detection_crop,
                "local_raw": local_raw_crop,
                "local_norm": local_norm_crop,
                "p_map": p_crop,
            }
        )
    else:
        y0, y1, x0, x1 = _roi_bbox(mask)
        mask_crop = mask[y0:y1, x0:x1]
        raw_display = base_norm[y0:y1, x0:x1]
        raw_display[~mask_crop] = 0.0
        detection_crop = (detection_mask & mask)[y0:y1, x0:x1]
        panel_entry.update(
            {
                "raw_display": raw_display,
                "mask_crop": mask_crop,
                "island_mask": detection_crop,
                "local_raw": None,
                "local_norm": None,
                "p_map": None,
            }
        )

    record = {
        "image_relpath": row.get("image_relpath"),
        "well": row.get("well"),
        "day": row.get("day"),
        "time": row.get("time"),
    }
    record.update(global_stats)
    record.update(local_results)
    return record, panel_entry



def _save_well_panel(
    prefix: str,
    well: str,
    entries: list[Dict[str, object]],
    panel_dir: Path,
) -> None:
    """Render a 6×N pipeline panel per well (columns = time points)."""

    filtered = [e for e in entries if e.get("raw_display") is not None]
    if not filtered:
        return

    filtered.sort(key=lambda e: (e.get("time_hours") if e.get("time_hours") is not None else float("inf")))
    n_time = len(filtered)

    local_vals = []
    for entry in filtered:
        arr = entry.get("local_raw")
        if arr is not None:
            vals = arr[np.isfinite(arr)]
            if vals.size:
                local_vals.append(vals)
    if local_vals:
        raw_min = min(v.min() for v in local_vals)
        raw_max = max(v.max() for v in local_vals)
        if raw_min == raw_max:
            raw_min -= 1.0
            raw_max += 1.0
    else:
        raw_min, raw_max = -1.0, 1.0
    norm_raw = colors.Normalize(vmin=raw_min, vmax=raw_max)

    fig_width = max(2.4 * n_time, 7.0)
    fig, axes = plt.subplots(6, n_time, figsize=(fig_width, 10.0))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(6, 1)

    cmap_gray = colormaps.get_cmap("gray")
    cmap_local = colormaps.get_cmap("viridis")
    last_local_raw_im = None
    last_local_scaled_im = None
    has_sig = False

    for idx, entry in enumerate(filtered):
        raw_display = entry.get("raw_display")
        mask_crop = entry.get("mask_crop")
        island_mask = entry.get("island_mask")
        local_raw = entry.get("local_raw")
        local_norm = entry.get("local_norm")
        p_map = entry.get("p_map")
        time_label = entry.get("time_label") or ""

        # Row 1: raw fluorescence (ROI-only grayscale)
        ax_raw = axes[0, idx]
        ax_raw.imshow(raw_display, cmap=cmap_gray, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        ax_raw.set_title(time_label, fontsize=11)

        # Row 2: ROI overlay (cyan)
        ax_roi = axes[1, idx]
        roi_rgb = np.repeat(raw_display[..., None], 3, axis=-1)
        roi_rgb[~mask_crop] = 0.0
        roi_rgb[mask_crop] = (1 - 0.6) * roi_rgb[mask_crop] + 0.6 * np.array([0.0, 1.0, 1.0])
        ax_roi.imshow(roi_rgb, interpolation="nearest")
        ax_roi.set_xticks([])
        ax_roi.set_yticks([])

        # Row 3: detected clusters (magenta overlay)
        ax_clusters = axes[2, idx]
        clusters_rgb = np.repeat(raw_display[..., None], 3, axis=-1)
        clusters_rgb[~mask_crop] = 0.0
        if island_mask is not None and island_mask.any():
            clusters_rgb[island_mask] = (1 - 0.6) * clusters_rgb[island_mask] + 0.6 * np.array([1.0, 0.0, 1.0])
        ax_clusters.imshow(clusters_rgb, interpolation="nearest")
        ax_clusters.set_xticks([])
        ax_clusters.set_yticks([])

        # Row 4: Local Moran raw values with actual scale
        ax_raw_moran = axes[3, idx]
        if local_raw is not None:
            local_raw_plot = np.where(mask_crop, local_raw, np.nan)
            last_local_raw_im = ax_raw_moran.imshow(local_raw_plot, cmap=cmap_local, norm=norm_raw, interpolation="nearest")
        else:
            last_local_raw_im = ax_raw_moran.imshow(np.zeros_like(raw_display), cmap=cmap_local, norm=norm_raw, interpolation="nearest")
        ax_raw_moran.set_xticks([])
        ax_raw_moran.set_yticks([])

        # Row 5: Local Moran scaled 0-1
        ax_scaled = axes[4, idx]
        if local_norm is not None:
            last_local_scaled_im = ax_scaled.imshow(np.where(mask_crop, local_norm, np.nan), cmap=cmap_local, vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            last_local_scaled_im = ax_scaled.imshow(np.zeros_like(raw_display), cmap=cmap_local, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_scaled.set_xticks([])
        ax_scaled.set_yticks([])

        # Row 6: significance overlay (p < 0.05)
        ax_sig = axes[5, idx]
        sig_rgb = np.repeat(raw_display[..., None], 3, axis=-1)
        sig_rgb[~mask_crop] = 0.0
        if p_map is not None and local_raw is not None:
            sig_mask = (p_map < 0.05) & mask_crop
            if np.any(sig_mask):
                has_sig = True
            color_vals = cmap_local(norm_raw(np.nan_to_num(local_raw, nan=raw_min)))[..., :3]
            sig_rgb[sig_mask] = (1 - 0.7) * sig_rgb[sig_mask] + 0.7 * color_vals[sig_mask]
        ax_sig.imshow(sig_rgb, interpolation="nearest")
        ax_sig.set_xticks([])
        ax_sig.set_yticks([])

    axes[0, 0].set_ylabel("Raw mCherry", fontsize=11)
    axes[1, 0].set_ylabel("ROI (cyan)", fontsize=11)
    axes[2, 0].set_ylabel("Detected clusters", fontsize=11)
    axes[3, 0].set_ylabel("Local Moran (raw)", fontsize=11)
    axes[4, 0].set_ylabel("Local Moran (scaled)", fontsize=11)
    axes[5, 0].set_ylabel("Significance (p<0.05)", fontsize=11)

    if last_local_raw_im is not None:
        cbar_raw = fig.colorbar(last_local_raw_im, ax=list(axes[3, :]), fraction=0.03, pad=0.02)
        cbar_raw.set_label("Local Moran (raw)")
    if last_local_scaled_im is not None:
        cbar_scaled = fig.colorbar(last_local_scaled_im, ax=list(axes[4, :]), fraction=0.03, pad=0.02)
        cbar_scaled.set_label("Local Moran (scaled 0-1)")
    if has_sig:
        sm_sig = cm.ScalarMappable(norm=norm_raw, cmap=cmap_local)
        cbar_sig = fig.colorbar(sm_sig, ax=list(axes[5, :]), fraction=0.03, pad=0.02)
        cbar_sig.set_label("Local Moran (significant, p<0.05)")

    fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.08, wspace=0.05, hspace=0.12)
    legend_text = (
        "Legend:\n"
        "cyan = ROI (analysis region)\n"
        "magenta = detected clusters\n"
        "heatmaps = Local Moran Ii\n"
        "row 6 color = p<0.05 (one-tailed, positive autocorr)"
    )
    fig.text(0.85, 0.5, legend_text, fontsize=8, va="center")
    fig.text(
        0.05,
        0.02,
        "Workflow visualization for a single well: raw fluorescence → ROI restriction → detected islands → Local Moran statistics (raw and display-scaled) → permutation-based significance map (p<0.05), highlighting spatially clustered regions within the analysis ROI.",
        fontsize=8,
    )

    panel_dir.mkdir(parents=True, exist_ok=True)
    base = panel_dir / f"{prefix}_well_{well}_moran_panel"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)


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
        "--panel-dir",
        type=Path,
        default=None,
        help="Directory to save per-well Moran panels (default: <project-root>/plots/morans/panels).",
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

    df = pd.read_csv(roi_path)
    if df.empty:
        raise SystemExit("roi_measurements.csv is empty.")

    if "day_index" not in df.columns:
        df["day_index"] = df["day"].apply(_parse_day)
    if "time_hours" not in df.columns:
        parsed_hours = df["time"].apply(_parse_time_to_hours)
        df["time_hours"] = (df["day_index"].fillna(0) * 24.0) + parsed_hours.fillna(0)

    heatmap_dir = (args.heatmap_dir or (default_output_dir / "heatmaps")).expanduser().resolve()
    overlay_dir = (args.overlay_dir or (default_output_dir / "overlays")).expanduser().resolve()
    pmap_dir = (heatmap_dir / "p_values").resolve()
    panel_dir = (args.panel_dir or (default_output_dir / "panels")).expanduser().resolve()
    for directory in (heatmap_dir, overlay_dir, pmap_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, object]] = []
    wells = list(df.groupby("well", sort=True))
    total_wells = len(wells)
    processed_rows = 0
    row_limit = args.limit if args.limit is not None else len(df)
    panel_prefix = output_csv.stem

    for w_idx, (well, group) in enumerate(wells, start=1):
        if processed_rows >= row_limit:
            break

        group = group.sort_values(["day_index", "time_hours"])
        print(f"\nProcessing well {well} ({w_idx}/{total_wells})")
        group_total = len(group)
        panel_entries: list[Dict[str, object]] = []

        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            if processed_rows >= row_limit:
                break
            time_label = row.get("time", "")
            bar_width = 28
            progress = idx / group_total if group_total else 1.0
            filled = int(bar_width * progress)
            bar = '#' * filled + '-' * (bar_width - filled)
            print(f"  [{bar}] {idx}/{group_total} time {time_label}", end='\r', flush=True)

            record, panel_entry = process_row(
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
            rows.append(record)
            panel_entries.append(panel_entry)
            processed_rows += 1

        print()
        _save_well_panel(panel_prefix, well, panel_entries, panel_dir)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Wrote Moran results for {len(result_df)} frames to {output_csv}")


if __name__ == "__main__":
    main()
