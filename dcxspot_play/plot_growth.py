"""Plot organoid ROI growth metrics from roi_measurements.csv."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib import colors
from matplotlib import colormaps

from .plotting import minimal_style_context
from .utils import apply_roi_mask, read_image, read_mask

CONFIG_PATH = Path(__file__).resolve().parent.parent / "dcxspot_config.json"


@dataclass
class ProcessedMeasurements:
    data: pd.DataFrame
    time_labels: pd.DataFrame
    div_start: Optional[int]


def _load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _default_project_root() -> Optional[Path]:
    cfg = _load_config()
    project_root = cfg.get("project_root")
    if not project_root:
        return None
    return Path(project_root).expanduser()



def _default_roi_measurements_path() -> Optional[Path]:
    project_root = _default_project_root()
    if project_root is None:
        return None
    candidate = project_root / "roi_measurements.csv"
    if candidate.exists():
        return candidate
    return None


def _parse_day(day_label: str) -> Optional[int]:
    if not isinstance(day_label, str):
        return None
    try:
        prefix, value = day_label.split("_", 1)
    except ValueError:
        return None
    if prefix.lower() != "day":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_time_to_hours(time_label: str) -> Optional[float]:
    if not isinstance(time_label, str):
        return None
    if "h" not in time_label or "m" not in time_label:
        return None
    hours_part, minutes_part = time_label.split("h", 1)
    minutes_part = minutes_part.rstrip("m")
    try:
        hours = int(hours_part)
        minutes = int(minutes_part) if minutes_part else 0
    except ValueError:
        return None
    return float(hours) + minutes / 60.0


def _format_time_label(day_index: int, time_label: str, div_start: Optional[int]) -> str:
    if div_start is not None:
        base = f"DIV{div_start + day_index}"
    else:
        base = f"day_{day_index:02d}"
    clean_time = time_label.strip() if isinstance(time_label, str) else ""
    if clean_time:
        return f"{base} {clean_time}"
    return base



def _build_time_labels(df: pd.DataFrame, div_start: Optional[int]) -> pd.DataFrame:
    labels = (
        df[["time_hours", "day_index", "time"]]
        .drop_duplicates()
        .sort_values("time_hours")
        .reset_index(drop=True)
    )
    labels["label"] = [
        _format_time_label(row.day_index, row.time, div_start)
        for row in labels.itertuples()
    ]
    if div_start is not None:
        labels["div_day"] = labels["day_index"] + div_start
    return labels


def _resolve_brightfield_path(row: pd.Series, project_root: Path) -> Path:
    rel = row.get('image_relpath')
    if isinstance(rel, str) and rel:
        candidate = (project_root / rel).expanduser().resolve()
        if candidate.exists():
            return candidate

    image_path = row.get('image_path')
    if isinstance(image_path, str):
        candidate = Path(image_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate image for row with relpath={rel!r} and image_path={row.get('image_path')!r}"
    )




def _prepare_measurements(csv_path: Path, div_start: Optional[int], project_root: Path) -> ProcessedMeasurements:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("roi_measurements.csv is empty")

    df = df.sort_values("timestamp_iso").drop_duplicates(subset=["image_relpath"], keep="last")

    df["day_index"] = df["day"].apply(_parse_day)
    df["time_in_hours"] = df["time"].apply(_parse_time_to_hours)

    if df["day_index"].isna().any() or df["time_in_hours"].isna().any():
        missing = df[df["day_index"].isna() | df["time_in_hours"].isna()][["image_relpath", "day", "time"]]
        raise ValueError(
            "Could not parse day/time for the following rows:\n"
            + missing.to_string(index=False)
        )

    project_root = project_root.expanduser().resolve()

    df["time_hours"] = df["day_index"] * 24.0 + df["time_in_hours"]
    df["time_days"] = df["time_hours"] / 24.0
    df = df.sort_values(["time_hours", "well"]).reset_index(drop=True)

    time_labels = _build_time_labels(df, div_start)

    if div_start is not None:
        df["div_day"] = df["day_index"] + div_start
        df["age_div"] = df["time_days"] + div_start

    brightfield_paths = [
        _resolve_brightfield_path(row, project_root) for _, row in df.iterrows()
    ]
    df["brightfield_path"] = brightfield_paths

    fluor_mean: list[float] = []
    fluor_sum: list[float] = []

    for path in brightfield_paths:
        mask_path = path.with_name(f"{path.stem}_mask.tif")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing ROI mask for {path}")

        fluor_path = path.parent / "fluorescence" / f"{path.stem}_mcherry.tif"
        if not fluor_path.exists():
            raise FileNotFoundError(f"Missing fluorescence image for {path}")

        mask = read_mask(mask_path)
        fluor_img = read_image(fluor_path).astype(np.float32, copy=False)
        fluor_masked = apply_roi_mask(fluor_img, mask, outside="nan")

        mean_val = float(np.nanmean(fluor_masked))
        sum_val = float(np.nansum(fluor_masked))
        fluor_mean.append(mean_val)
        fluor_sum.append(sum_val)

    df["fluor_density_intensity"] = fluor_mean
    df["fluor_sum_intensity"] = fluor_sum

    per_well = df.sort_values(["well", "time_hours"])
    baseline_area = per_well.groupby("well")["area_px"].transform("first")
    baseline_fluor_sum = per_well.groupby("well")["fluor_sum_intensity"].transform("first")
    baseline_fluor_density = per_well.groupby("well")["fluor_density_intensity"].transform("first")

    per_well = per_well.assign(
        baseline_area_px=baseline_area,
        area_fold_change=lambda x: np.where(
            x["baseline_area_px"] > 0,
            x["area_px"] / x["baseline_area_px"],
            np.nan,
        ),
        baseline_fluor_sum=baseline_fluor_sum,
        baseline_fluor_density=baseline_fluor_density,
    )
    per_well["fluor_sum_fold_change"] = np.where(
        per_well["baseline_fluor_sum"] > 0,
        per_well["fluor_sum_intensity"] / per_well["baseline_fluor_sum"],
        np.nan,
    )
    per_well["fluor_density_fold_change"] = np.where(
        per_well["baseline_fluor_density"] > 0,
        per_well["fluor_density_intensity"] / per_well["baseline_fluor_density"],
        np.nan,
    )

    df.loc[per_well.index, "baseline_area_px"] = per_well["baseline_area_px"].values
    df.loc[per_well.index, "area_fold_change"] = per_well["area_fold_change"].values
    df.loc[per_well.index, "baseline_fluor_sum"] = per_well["baseline_fluor_sum"].values
    df.loc[per_well.index, "baseline_fluor_density"] = per_well["baseline_fluor_density"].values
    df.loc[per_well.index, "fluor_sum_fold_change"] = per_well["fluor_sum_fold_change"].values
    df.loc[per_well.index, "fluor_density_fold_change"] = per_well["fluor_density_fold_change"].values

    return ProcessedMeasurements(
        data=df,
        time_labels=time_labels,
        div_start=div_start,
    )



def _save_multi_format(fig: plt.Figure, output_base: Path) -> List[Path]:
    saved_paths: List[Path] = []
    for ext in ('.png', '.pdf', '.svg'):
        out_path = output_base.with_suffix(ext)
        save_kwargs = {'dpi': 300} if ext == '.png' else {}
        fig.savefig(out_path, **save_kwargs)
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths



def _plot_metric_boxplot(
    processed: ProcessedMeasurements,
    value_column: str,
    output_base: Path,
    ylabel: str,
    title: str,
    y_min: Optional[float] = None,
) -> List[Path]:
    df = processed.data
    labels_df = processed.time_labels

    data = []
    tick_labels = []
    for row in labels_df.itertuples():
        values = df.loc[df['time_hours'] == row.time_hours, value_column].dropna()
        if values.empty:
            continue
        data.append(values.values)
        tick_labels.append(row.label)

    if not data:
        raise ValueError(f'No measurements available for {value_column}')

    positions = np.arange(len(data)) + 1

    with minimal_style_context():
        fig, ax = plt.subplots()
        boxprops = dict(linewidth=1.0, facecolor='#1f77b4', alpha=0.25)
        medianprops = dict(color='#1f77b4', linewidth=1.5)
        whiskerprops = dict(color='#8c8c8c', linewidth=1.0)
        capprops = dict(color='#8c8c8c', linewidth=1.0)
        flierprops = dict(
            marker='o',
            markersize=3,
            markerfacecolor='#1f77b4',
            markeredgecolor='none',
            alpha=0.4,
        )

        ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlim(0.5, len(data) + 0.5)
        xlabel = 'Organoid age (DIV)' if processed.div_start is not None else 'Time point'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        saved = _save_multi_format(fig, output_base)

    return saved



def _plot_area_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='area_px',
        output_base=output_base,
        ylabel='Organoid ROI area (pixels)',
        title='Organoid area per timepoint',
    )



def _plot_growth_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='area_fold_change',
        output_base=output_base,
        ylabel='Fold-change in area (vs first time-point)',
        title='Organoid growth per timepoint',
        y_min=0.0,
    )


def _plot_fluor_total_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='fluor_sum_intensity',
        output_base=output_base,
        ylabel='Total mCherry fluorescence (a.u.)',
        title='Total fluorescence per timepoint',
    )



def _plot_fluor_total_growth_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='fluor_sum_fold_change',
        output_base=output_base,
        ylabel='Total fluorescence fold-change (vs first time-point)',
        title='Total fluorescence growth per timepoint',
        y_min=0.0,
    )



def _plot_fluor_density_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='fluor_density_intensity',
        output_base=output_base,
        ylabel='Fluorescence per area (a.u. per pixel)',
        title='Area-normalised fluorescence per timepoint',
    )



def _plot_fluor_density_growth_boxplot(processed: ProcessedMeasurements, output_base: Path) -> List[Path]:
    return _plot_metric_boxplot(
        processed,
        value_column='fluor_density_fold_change',
        output_base=output_base,
        ylabel='Area-normalised fluorescence fold-change',
        title='Area-normalised fluorescence growth per timepoint',
        y_min=0.0,
    )




def _normalize_display_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return arr
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(valid, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _crop_to_mask(image: np.ndarray, mask: np.ndarray, pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    mask_bool = mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return image, mask_bool
    y0 = max(0, ys.min() - pad)
    y1 = min(mask_bool.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask_bool.shape[1], xs.max() + pad + 1)
    if image.ndim > 2:
        cropped_image = image[..., y0:y1, x0:x1]
    elif image.ndim == 2:
        cropped_image = image[y0:y1, x0:x1]
    else:
        cropped_image = image
    cropped_mask = mask_bool[y0:y1, x0:x1]
    return cropped_image, cropped_mask



def _generate_well_panels(
    processed: ProcessedMeasurements,
    project_root: Path,
    output_dir: Path,
    prefix: str,
) -> List[Path]:
    saved_paths: List[Path] = []
    df = processed.data
    label_lookup = processed.time_labels.set_index('time_hours')['label']

    gray_cmap = colormaps['gray'].with_extremes(bad='black')
    magma_cmap = colormaps['magma'].with_extremes(bad='black')

    for well, per_well in df.groupby('well', sort=True):
        per_well = per_well.sort_values('time_hours')
        entries: List[dict] = []
        raw_values: List[np.ndarray] = []

        for row in per_well.itertuples():
            brightfield_path = Path(row.brightfield_path)
            mask_path = brightfield_path.with_name(f"{brightfield_path.stem}_mask.tif")
            fluor_path = brightfield_path.parent / 'fluorescence' / f"{brightfield_path.stem}_mcherry.tif"

            mask = read_mask(mask_path)
            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                continue

            brightfield_img = read_image(brightfield_path).astype(np.float32, copy=False)
            fluor_img = read_image(fluor_path).astype(np.float32, copy=False)

            bf_crop, mask_crop = _crop_to_mask(brightfield_img, mask_bool)
            fluor_crop, _ = _crop_to_mask(fluor_img, mask_bool)
            fluor_masked = apply_roi_mask(fluor_img, mask_bool, outside='zero')
            fluor_masked_crop, _ = _crop_to_mask(fluor_masked, mask_bool)

            bf_roi = np.where(mask_crop, bf_crop, np.nan)
            fluor_roi = np.where(mask_crop, fluor_crop, np.nan)
            fluor_masked_roi = np.where(mask_crop, fluor_masked_crop, np.nan)

            entries.append(
                {
                    'time_hours': row.time_hours,
                    'mask': mask_crop,
                    'bf_roi': bf_roi,
                    'fluor_roi': fluor_roi,
                    'fluor_masked_roi': fluor_masked_roi,
                }
            )
            raw_values.append(fluor_roi[mask_crop])

        if not entries:
            continue

        raw_concat = np.concatenate(raw_values)
        if raw_concat.size == 0:
            continue
        raw_min = float(np.nanmin(raw_concat))
        raw_max = float(np.nanmax(raw_concat))
        if not np.isfinite(raw_min) or not np.isfinite(raw_max) or raw_max <= raw_min:
            raw_min, raw_max = 0.0, 1.0
        raw_norm = colors.Normalize(vmin=raw_min, vmax=raw_max)
        scaled_norm = colors.Normalize(vmin=0.0, vmax=1.0)

        entries.sort(key=lambda x: x['time_hours'])
        n_time = len(entries)

        with minimal_style_context():
            fig_width = max(2.2 * n_time, 6.0)
            fig, axes = plt.subplots(3, n_time, figsize=(fig_width, 6.4))
            axes = np.asarray(axes)
            if axes.ndim == 1:
                axes = axes.reshape(3, 1)

            for idx, entry in enumerate(entries):
                time_label = label_lookup.get(entry['time_hours'], f"{entry['time_hours']:.1f} h")

                bf_norm = _normalize_display_image(entry['bf_roi'])
                ax_bf = axes[0, idx]
                ax_bf.imshow(bf_norm, cmap=gray_cmap, interpolation='nearest', vmin=0.0, vmax=1.0)
                ax_bf.set_xticks([])
                ax_bf.set_yticks([])
                ax_bf.set_title(time_label, fontsize=11)

                ax_raw = axes[1, idx]
                ax_raw.imshow(entry['fluor_roi'], cmap=magma_cmap, norm=raw_norm, interpolation='nearest')
                ax_raw.set_xticks([])
                ax_raw.set_yticks([])

                scaled = _normalize_display_image(entry['fluor_masked_roi'])
                ax_scaled = axes[2, idx]
                ax_scaled.imshow(scaled, cmap=magma_cmap, norm=scaled_norm, interpolation='nearest')
                ax_scaled.set_xticks([])
                ax_scaled.set_yticks([])

            axes[0, 0].set_ylabel('Brightfield (ROI)', fontsize=11)
            axes[1, 0].set_ylabel('mCherry (raw ROI)', fontsize=11)
            axes[2, 0].set_ylabel('mCherry (scaled ROI)', fontsize=11)

            fig.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.06, wspace=0.05, hspace=0.08)

            sm_raw = cm.ScalarMappable(norm=raw_norm, cmap=magma_cmap)
            cbar_raw = fig.colorbar(sm_raw, ax=list(axes[1, :]), orientation='vertical', fraction=0.035, pad=0.02)
            cbar_raw.set_label('Intensity (a.u.)')

            sm_scaled = cm.ScalarMappable(norm=scaled_norm, cmap=magma_cmap)
            cbar_scaled = fig.colorbar(sm_scaled, ax=list(axes[2, :]), orientation='vertical', fraction=0.035, pad=0.02)
            cbar_scaled.set_label('Scaled intensity (0-1)')

            fig.suptitle(f"Well {well}", x=0.05, ha='left', y=0.96, fontsize=14)

            output_base = output_dir / f"{prefix}_well_{well}_panel"
            saved_paths.extend(_save_multi_format(fig, output_base))

    return saved_paths





def parse_args() -> argparse.Namespace:
    default_input = _default_roi_measurements_path()
    cfg = _load_config()

    parser = argparse.ArgumentParser(description="Plot organoid ROI size and growth box plots with a consistent style.")
    parser.add_argument(
        "--roi-measurements",
        type=Path,
        default=default_input,
        required=default_input is None,
        help="Path to roi_measurements.csv. Defaults to the project root defined in dcxspot_config.json if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be written. Defaults to <project-root>/plots.",
    )
    parser.add_argument(
        "--div-start",
        type=int,
        default=cfg.get("div_start"),
        help="DIV value corresponding to day_00 (e.g., 11 for DIV11). Defaults to dcxspot_config.json if present.",
    )

    parser.add_argument(
        "--prefix",
        default="organoid",
        help="Filename prefix for generated plots (default: organoid).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.roi_measurements.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"ROI measurements file not found: {input_path}")

    project_root = _default_project_root()
    if project_root is None:
        project_root = input_path.parent
    project_root = project_root.expanduser().resolve()

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = (project_root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = _prepare_measurements(input_path, args.div_start, project_root)

    area_base = output_dir / f"{args.prefix}_area_boxplot"
    growth_base = output_dir / f"{args.prefix}_growth_boxplot"
    fluor_total_base = output_dir / f"{args.prefix}_fluor_total_boxplot"
    fluor_total_growth_base = output_dir / f"{args.prefix}_fluor_total_growth_boxplot"
    fluor_density_base = output_dir / f"{args.prefix}_fluor_density_boxplot"
    fluor_density_growth_base = output_dir / f"{args.prefix}_fluor_density_growth_boxplot"

    area_paths = _plot_area_boxplot(processed, area_base)
    growth_paths = _plot_growth_boxplot(processed, growth_base)
    fluor_total_paths = _plot_fluor_total_boxplot(processed, fluor_total_base)
    fluor_total_growth_paths = _plot_fluor_total_growth_boxplot(processed, fluor_total_growth_base)
    fluor_density_paths = _plot_fluor_density_boxplot(processed, fluor_density_base)
    fluor_density_growth_paths = _plot_fluor_density_growth_boxplot(processed, fluor_density_growth_base)
    panel_paths = _generate_well_panels(processed, project_root, output_dir, args.prefix)

    for path in area_paths:
        print(f"Saved area box plot to {path}")
    for path in growth_paths:
        print(f"Saved growth box plot to {path}")
    for path in fluor_total_paths:
        print(f"Saved total fluorescence box plot to {path}")
    for path in fluor_total_growth_paths:
        print(f"Saved total fluorescence growth box plot to {path}")
    for path in fluor_density_paths:
        print(f"Saved area-normalised fluorescence box plot to {path}")
    for path in fluor_density_growth_paths:
        print(f"Saved area-normalised fluorescence growth box plot to {path}")
    for path in panel_paths:
        if Path(path).suffix == '.png':
            print(f"Saved well panel to {path}")


if __name__ == "__main__":
    main()
