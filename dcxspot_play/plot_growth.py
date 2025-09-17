"""Plot organoid ROI growth metrics from roi_measurements.csv."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from .plotting import minimal_style_context

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


def _prepare_measurements(csv_path: Path, div_start: Optional[int]) -> ProcessedMeasurements:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("roi_measurements.csv is empty")

    df = df.sort_values("timestamp_iso").drop_duplicates(subset=["image_relpath"], keep="last")

    df["day_index"] = df["day"].apply(_parse_day)
    df["time_in_hours"] = df["time"].apply(_parse_time_to_hours)

    if df["day_index"].isna().any() or df["time_in_hours"].isna().any():
        missing = df[df["day_index"].isna() | df["time_in_hours"].isna()][["image_relpath", "day", "time"]]
        raise ValueError(
            "Could not parse day/time for the following rows:\n" + missing.to_string(index=False)
        )

    df["time_hours"] = df["day_index"] * 24.0 + df["time_in_hours"]
    df["time_days"] = df["time_hours"] / 24.0
    df = df.sort_values(["time_hours", "well"]).reset_index(drop=True)

    time_labels = _build_time_labels(df, div_start)

    if div_start is not None:
        df["div_day"] = df["day_index"] + div_start
        df["age_div"] = df["time_days"] + div_start

    per_well = df.sort_values(["well", "time_hours"])
    baseline = per_well.groupby("well")["area_px"].transform("first")
    per_well = per_well.assign(
        baseline_area_px=baseline,
        area_fold_change=lambda x: np.where(x["baseline_area_px"] > 0, x["area_px"] / x["baseline_area_px"], np.nan),
    )
    df.loc[per_well.index, "baseline_area_px"] = per_well["baseline_area_px"].values
    df.loc[per_well.index, "area_fold_change"] = per_well["area_fold_change"].values

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

    processed = _prepare_measurements(input_path, args.div_start)

    area_base = output_dir / f"{args.prefix}_area_boxplot"
    growth_base = output_dir / f"{args.prefix}_growth_boxplot"

    area_paths = _plot_area_boxplot(processed, area_base)
    growth_paths = _plot_growth_boxplot(processed, growth_base)

    for path in area_paths:
        print(f"Saved area box plot to {path}")
    for path in growth_paths:
        print(f"Saved growth box plot to {path}")


if __name__ == "__main__":
    main()
