"""Batch runner for Moran's I spatial autocorrelation across ROI snapshots.

This module expects each ROI entry in ``roi_measurements.csv`` to have the
standard artefacts written by the Organoid ROI + DCX pipelines:

* Brightfield image (``<stem>.tif``)
* ROI mask (``<stem>_mask.tif``)
* Raw mCherry fluorescence (``fluorescence/<stem>_mcherry.tif``)
* Optional DCX detection labels (``dcxspot/<stem>_labels.tif``)

For every time point we compute global Moran's I (with permutations) on the raw
ROI fluorescence, derive local Moran statistics (raw Ii values and
permutation-based p-values), and emit the following deliverables:

* CSV summary of global Moran's I for every ROI.
* Local Moran heatmaps, overlays, and p-value maps (one per ROI).
* A per-well, 6-row montage illustrating the full processing workflow—raw
  fluorescence, ROI mask, detected islands, raw Local Moran map, scaled Local
  Moran map, and significance mask—so the provenance of the statistics is
  visually explicit.

All numerical computations use the raw fluorescence intensities inside the ROI
after mean-centering; any percentile-based scaling is limited to figure
generation for visual clarity.
"""
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
    """Prepare display-ready arrays for the Local Moran overlay pipeline."""
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
    overlay_dir: Optional[Path],
    heatmap_dir: Optional[Path],
    pmap_dir: Optional[Path],
    panel_requested: bool,
    random_state: Optional[int],
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    """Compute Moran’s I statistics for a single ROI and collect panel data."""

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

    time_str = row.get("time", "") or ""
    div_day = row.get("div_day")
    if pd.notna(div_day):
        time_str = f"DIV{int(div_day)} {time_str}"

    panel_entry: Optional[Dict[str, object]] = None
    local_results: Dict[str, Path] = {}

    # detection labels for ROI overlay and cluster counts
    detection_labels = load_detection_labels(brightfield_path)
    if detection_labels is not None and detection_labels.shape == mask.shape:
        detection_mask = detection_labels.astype(bool)
        labels_roi = detection_labels[mask]
        cluster_count = int(np.unique(labels_roi[labels_roi > 0]).size)
    else:
        detection_mask = np.zeros_like(mask, dtype=bool)
        cluster_count = 0

    # Determine whether we need to render any figures/panels
    need_heatmap = heatmap_dir is not None
    need_overlay = overlay_dir is not None
    need_pmap = pmap_dir is not None and local_permutations > 0
    panel_needed = panel_requested
    visuals_needed = need_heatmap or need_overlay or need_pmap or panel_needed

    bf_display = _normalize_for_display(brightfield, mask) if panel_needed else None
    raw_display = _normalize_for_display(fluor_masked, mask) if (visuals_needed or panel_needed) else None

    if panel_needed and bf_display is not None and raw_display is not None:
        y0, y1, x0, x1 = _roi_bbox(mask)
        mask_crop = mask[y0:y1, x0:x1]
        bf_crop = np.where(mask_crop, bf_display[y0:y1, x0:x1], 0.0)
        raw_crop = np.where(mask_crop, raw_display[y0:y1, x0:x1], 0.0)
        island_crop = (detection_mask & mask)[y0:y1, x0:x1]
        panel_entry = {
            "time_label": time_str.strip(),
            "time_hours": row.get("time_hours"),
            "bf_display": bf_crop,
            "raw_display": raw_crop,
            "mask_crop": mask_crop,
            "island_mask": island_crop,
            "cluster_count": cluster_count,
        }

    local_map = None
    local_norm = None
    p_map = None

    if visuals_needed or local_permutations > 0:
        _, _, _, _, _, xc, neighbor_sum_xc, _ = morans_i_snapshot(
            fluor_masked,
            mask,
            neighbors=neighbors,
        )
        local_map = local_moran_map(xc, mask, neighbor_sum_xc)

        if need_heatmap:
            heatmap_path = heatmap_dir / f"{base_name}_local_moran.png"
            _save_heatmap(local_map, mask, heatmap_path, "Local Moran (Ii)")
            local_results["local_heatmap"] = heatmap_path

        if need_overlay:
            _, local_norm, _, overlay_rgb = _render_overlay_arrays(
                fluor_masked,
                mask,
                local_map,
            )
            overlay_path = overlay_dir / f"{base_name}_local_overlay.png"
            plt.imsave(overlay_path, overlay_rgb)
            local_results["local_overlay"] = overlay_path

        if need_pmap:
            Ii_map, p_map = local_moran_permutation_pvals(
                fluor_masked,
                mask,
                neighbors=neighbors,
                local_permutations=local_permutations,
                random_state=random_state,
            )
            local_map = Ii_map
            p_path = pmap_dir / f"{base_name}_local_pvals.png"
            _save_heatmap(p_map, mask, p_path, "Local Moran p-values")
            local_results["local_pmap"] = p_path

    if panel_entry is not None:
        y0, y1, x0, x1 = _roi_bbox(mask)
        if local_map is not None:
            local_map = np.where(mask, local_map, np.nan)
            panel_entry["local_raw"] = local_map[y0:y1, x0:x1]
        else:
            panel_entry["local_raw"] = None
        if local_norm is not None:
            local_norm = np.where(mask, local_norm, np.nan)
            panel_entry["local_norm"] = local_norm[y0:y1, x0:x1]
        else:
            panel_entry["local_norm"] = None
        if p_map is not None:
            p_map = np.where(mask, p_map, np.nan)
            panel_entry["p_map"] = p_map[y0:y1, x0:x1]
        else:
            panel_entry["p_map"] = None

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
    """Render the multi-row workflow panel for a single well."""

    filtered = [e for e in entries if e is not None and e.get("raw_display") is not None]
    if not filtered:
        return

    filtered.sort(
        key=lambda e: (
            e.get("time_hours") if e.get("time_hours") is not None else float("inf")
        )
    )
    n_time = len(filtered)

    local_vals: list[np.ndarray] = []
    for entry in filtered:
        arr = entry.get("local_raw")
        if arr is not None:
            vals = arr[np.isfinite(arr)]
            if vals.size:
                local_vals.append(vals)
    if local_vals:
        all_vals = np.concatenate(local_vals)
        mean = float(np.nanmean(all_vals))
        std = float(np.nanstd(all_vals))
        clip = std * 2.0 if std > 0 else float(np.nanmax(np.abs(all_vals)))
        clip = clip if clip > 0 else 1.0
        vmin = mean - clip
        vmax = mean + clip
    else:
        vmin, vmax = -1.0, 1.0
    norm_raw = colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    fig_width = max(2.5 * n_time, 8.0)
    fig, axes = plt.subplots(7, n_time, figsize=(fig_width, 12.0))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(7, 1)

    cmap_gray = colormaps.get_cmap("gray")
    cmap_local = colormaps.get_cmap("coolwarm")
    cmap_scaled = colormaps.get_cmap("viridis")

    last_local_raw_im = None
    last_local_scaled_im = None
    last_sig_im = None
    has_sig = False

    for idx, entry in enumerate(filtered):
        bf_display = entry.get("bf_display")
        raw_display = entry.get("raw_display")
        mask_crop = entry.get("mask_crop")
        island_mask = entry.get("island_mask")
        local_raw = entry.get("local_raw")
        local_norm = entry.get("local_norm")
        p_map = entry.get("p_map")
        cluster_count = entry.get("cluster_count", 0)
        time_label = entry.get("time_label") or ""

        ax_bf = axes[0, idx]
        ax_bf.imshow(bf_display, cmap=cmap_gray, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_bf.set_xticks([])
        ax_bf.set_yticks([])
        ax_bf.set_title(time_label, fontsize=11)

        ax_raw = axes[1, idx]
        ax_raw.imshow(raw_display, cmap=cmap_gray, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])

        ax_roi = axes[2, idx]
        roi_rgb = np.repeat(raw_display[..., None], 3, axis=-1)
        roi_rgb[~mask_crop] = 0.0
        roi_rgb[mask_crop] = (1 - 0.6) * roi_rgb[mask_crop] + 0.6 * np.array([0.0, 1.0, 1.0])
        ax_roi.imshow(roi_rgb, interpolation="nearest")
        ax_roi.set_xticks([])
        ax_roi.set_yticks([])

        ax_clusters = axes[3, idx]
        clusters_rgb = np.repeat(raw_display[..., None], 3, axis=-1)
        clusters_rgb[~mask_crop] = 0.0
        if island_mask is not None and island_mask.any():
            clusters_rgb[island_mask] = (1 - 0.6) * clusters_rgb[island_mask] + 0.6 * np.array([1.0, 0.0, 1.0])
            ax_clusters.contour(island_mask.astype(float), levels=[0.5], colors="#ff00ff", linewidths=0.8)
        ax_clusters.imshow(clusters_rgb, interpolation="nearest")
        ax_clusters.text(
            0.02,
            0.95,
            f"{cluster_count} islands",
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax_clusters.transAxes,
            bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"),
        )
        ax_clusters.set_xticks([])
        ax_clusters.set_yticks([])

        ax_raw_moran = axes[4, idx]
        if local_raw is not None:
            local_raw_plot = np.where(mask_crop, local_raw, np.nan)
            last_local_raw_im = ax_raw_moran.imshow(local_raw_plot, cmap=cmap_local, norm=norm_raw, interpolation="nearest")
        else:
            last_local_raw_im = ax_raw_moran.imshow(np.zeros_like(raw_display), cmap=cmap_local, norm=norm_raw, interpolation="nearest")
        ax_raw_moran.set_xticks([])
        ax_raw_moran.set_yticks([])

        ax_scaled = axes[5, idx]
        if local_norm is not None:
            last_local_scaled_im = ax_scaled.imshow(np.where(mask_crop, local_norm, np.nan), cmap=cmap_scaled, vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            last_local_scaled_im = ax_scaled.imshow(np.zeros_like(raw_display), cmap=cmap_scaled, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_scaled.set_xticks([])
        ax_scaled.set_yticks([])

        ax_sig = axes[6, idx]
        ax_sig.imshow(raw_display, cmap=cmap_gray, vmin=0.0, vmax=1.0, interpolation="nearest")
        if p_map is not None and local_raw is not None:
            sig_mask = (p_map < 0.05) & mask_crop
            if np.any(sig_mask):
                has_sig = True
            sig_vals = np.where(sig_mask, local_raw, np.nan)
            sig_im = ax_sig.imshow(sig_vals, cmap=cmap_local, norm=norm_raw, interpolation="nearest")
            sig_im.set_alpha(np.where(np.isfinite(sig_vals), 0.85, 0.0))
            last_sig_im = sig_im
        ax_sig.set_xticks([])
        ax_sig.set_yticks([])

    row_labels = [
        "Brightfield",
        "Raw mCherry",
        "ROI (cyan)",
        "Detected clusters",
        "Local Moran (raw)",
        "Local Moran (scaled)",
        "Significance (p<0.05)",
    ]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    if last_local_raw_im is not None:
        cbar_raw = fig.colorbar(last_local_raw_im, ax=list(axes[4, :]), fraction=0.03, pad=0.02)
        cbar_raw.set_label("Local Moran (raw)")
    if last_local_scaled_im is not None:
        cbar_scaled = fig.colorbar(last_local_scaled_im, ax=list(axes[5, :]), fraction=0.03, pad=0.02)
        cbar_scaled.set_label("Local Moran (scaled 0-1)")
    if has_sig and last_sig_im is not None:
        cbar_sig = fig.colorbar(last_sig_im, ax=list(axes[6, :]), fraction=0.03, pad=0.02)
        cbar_sig.set_label("Local Moran (significant, p<0.05)")

    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.08, wspace=0.05, hspace=0.12)
    legend_text = (
        "Legend:
"
        "cyan = ROI (analysis region)
"
        "magenta = detected clusters
"
        "coolwarm heatmaps = Local Moran Ii
"
        "row 7 colors = Local Moran for p<0.05 (one-tailed, positive autocorr)"
    )
    fig.text(0.85, 0.5, legend_text, fontsize=8, va="center")
    fig.text(
        0.06,
        0.02,
        "Workflow visualization for a single well: brightfield → raw fluorescence → ROI restriction → detected islands → Local Moran statistics (raw and display-scaled) → permutation-based significance map (p<0.05).",
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
    cfg = _load_config()

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
        "--analysis-only",
        action="store_true",
        help="Skip figure generation; only write the global Moran CSV.",
    )
    parser.add_argument(
        "--wells",
        nargs="*",
        help="Optional list of wells to process (e.g. H11 H12). Case-insensitive.",
    )
    parser.add_argument(
        "--div-start",
        type=int,
        default=cfg.get("div_start"),
        help="DIV value aligned with day_00 (default: value from dcxspot_config.json if present).",
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

    analysis_only = args.analysis_only

    if not analysis_only:
        heatmap_dir = (args.heatmap_dir or (default_output_dir / "heatmaps")).expanduser().resolve()
        overlay_dir = (args.overlay_dir or (default_output_dir / "overlays")).expanduser().resolve()
        panel_dir = (args.panel_dir or (default_output_dir / "panels")).expanduser().resolve()
        pmap_dir = (heatmap_dir / "p_values").resolve()
        for directory in (heatmap_dir, overlay_dir, panel_dir, pmap_dir):
            directory.mkdir(parents=True, exist_ok=True)
    else:
        heatmap_dir = overlay_dir = panel_dir = pmap_dir = None

    df = pd.read_csv(roi_path)
    if df.empty:
        raise SystemExit("roi_measurements.csv is empty.")

    if args.wells:
        wells_upper = {w.upper() for w in args.wells}
        df = df[df["well"].str.upper().isin(wells_upper)]
        if df.empty:
            raise SystemExit(f"No rows found in {roi_path} for wells: {', '.join(args.wells)}")

    if "day_index" not in df.columns:
        df["day_index"] = df["day"].apply(_parse_day)
    if "time_hours" not in df.columns:
        parsed_hours = df["time"].apply(_parse_time_to_hours)
        df["time_hours"] = (df["day_index"].fillna(0) * 24.0) + parsed_hours.fillna(0)

    if args.div_start is not None:
        df["div_day"] = df["day_index"].fillna(0) + args.div_start

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
