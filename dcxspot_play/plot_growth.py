"""Plot organoid ROI growth metrics from roi_measurements.csv."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from .plotting import minimal_style_context

CONFIG_PATH = Path(__file__).resolve().parent.parent / "dcxspot_config.json"


@dataclass
class ProcessedMeasurements:
    data: pd.DataFrame
    area_summary: pd.DataFrame
    growth_summary: pd.DataFrame
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

    per_well = df.sort_values(["well", "time_hours"])  # ensure chronological order per well
    baseline = per_well.groupby("well")["area_px"].transform("first")
    per_well = per_well.assign(
        baseline_area_px=baseline,
        area_fold_change=lambda x: np.where(x["baseline_area_px"] > 0, x["area_px"] / x["baseline_area_px"], np.nan),
    )
    df.loc[per_well.index, "baseline_area_px"] = per_well["baseline_area_px"].values
    df.loc[per_well.index, "area_fold_change"] = per_well["area_fold_change"].values

    area_summary = (
        df.groupby("time_hours")
        .agg(
            median_area_px=("area_px", "median"),
            mean_area_px=("area_px", "mean"),
            q1_area_px=("area_px", lambda s: s.quantile(0.25)),
            q3_area_px=("area_px", lambda s: s.quantile(0.75)),
            count=("area_px", "count"),
        )
        .reset_index()
        .sort_values("time_hours")
    )

    area_summary = area_summary.merge(time_labels[["time_hours", "label"]], on="time_hours", how="left")

    growth_summary = (
        df.dropna(subset=["area_fold_change"])
        .groupby("time_hours")
        .agg(
            mean_fold=("area_fold_change", "mean"),
            median_fold=("area_fold_change", "median"),
            std_fold=("area_fold_change", "std"),
            count=("area_fold_change", "count"),
        )
        .reset_index()
        .sort_values("time_hours")
    )
    growth_summary = growth_summary.merge(time_labels[["time_hours", "label"]], on="time_hours", how="left")
    growth_summary["sem_fold"] = growth_summary.apply(
        lambda row: row["std_fold"] / math.sqrt(row["count"]) if row["count"] > 1 and not math.isnan(row["std_fold"]) else 0.0,
        axis=1,
    )

    return ProcessedMeasurements(
        data=df,
        area_summary=area_summary,
        growth_summary=growth_summary,
        time_labels=time_labels,
        div_start=div_start,
    )


def _plot_area_over_time(processed: ProcessedMeasurements, output_path: Path) -> None:
    df = processed.data
    summary = processed.area_summary
    labels_df = processed.time_labels

    with minimal_style_context():
        fig, ax = plt.subplots()

        # Light per-well trajectories to visualize variability without clutter.
        for _, per_well in df.groupby("well", sort=False):
            per_well = per_well.sort_values("time_hours")
            ax.plot(
                per_well["time_hours"],
                per_well["area_px"],
                color="#b7b7b7",
                linewidth=0.8,
                alpha=0.4,
            )

        ax.plot(
            summary["time_hours"],
            summary["median_area_px"],
            color="#1f77b4",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Median across wells",
        )
        ax.fill_between(
            summary["time_hours"],
            summary["q1_area_px"],
            summary["q3_area_px"],
            color="#1f77b4",
            alpha=0.12,
            label="Interquartile range",
        )

        if not labels_df.empty:
            ax.set_xticks(labels_df["time_hours"])
            ax.set_xticklabels(labels_df["label"], rotation=45, ha="right")

        if processed.div_start is not None:
            ax.set_xlabel("Organoid age (DIV)")
        else:
            ax.set_xlabel("Time (hours post start)")
        ax.set_ylabel("Organoid ROI area (pixels)")
        ax.set_title("Organoid area over time")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def _plot_growth_curve(processed: ProcessedMeasurements, output_path: Path) -> None:
    df = processed.data
    summary = processed.growth_summary
    labels_df = processed.time_labels

    with minimal_style_context():
        fig, ax = plt.subplots()

        for _, per_well in df.groupby("well", sort=False):
            per_well = per_well.dropna(subset=["area_fold_change"])  # skip wells without baseline
            if per_well.empty:
                continue
            per_well = per_well.sort_values("time_hours")
            ax.plot(
                per_well["time_hours"],
                per_well["area_fold_change"],
                color="#b7b7b7",
                linewidth=0.8,
                alpha=0.35,
            )

        if not summary.empty:
            ax.plot(
                summary["time_hours"],
                summary["mean_fold"],
                color="#2ca02c",
                linewidth=2.1,
                marker="o",
                markersize=4,
                label="Mean fold-change",
            )
            ax.fill_between(
                summary["time_hours"],
                summary["mean_fold"] - summary["sem_fold"],
                summary["mean_fold"] + summary["sem_fold"],
                color="#2ca02c",
                alpha=0.15,
                label="Mean ± SEM",
            )

        ax.axhline(1.0, color="#4a4a4a", linestyle="--", linewidth=1.0, alpha=0.7)
        if not labels_df.empty:
            ax.set_xticks(labels_df["time_hours"])
            ax.set_xticklabels(labels_df["label"], rotation=45, ha="right")
        if processed.div_start is not None:
            ax.set_xlabel("Organoid age (DIV)")
        else:
            ax.set_xlabel("Time (hours post start)")
        ax.set_ylabel("Fold-change in area (relative to first time-point)")
        ax.set_title("Organoid growth curve")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)



def _plot_area_boxplots(processed: ProcessedMeasurements, output_path: Path) -> None:
    df = processed.data
    labels_df = processed.time_labels

    if labels_df.empty:
        raise ValueError("No time labels found for box plot")

    data = []
    tick_labels = []
    for row in labels_df.itertuples():
        values = df.loc[df["time_hours"] == row.time_hours, "area_px"].dropna()
        if values.empty:
            continue
        data.append(values.values)
        tick_labels.append(row.label)

    if not data:
        raise ValueError("No area measurements available to plot")

    positions = np.arange(len(data))

    with minimal_style_context():
        fig, ax = plt.subplots()
        boxprops = dict(linewidth=1.0, facecolor="#1f77b4", alpha=0.25)
        medianprops = dict(color="#1f77b4", linewidth=1.5)
        whiskerprops = dict(color="#8c8c8c", linewidth=1.0)
        capprops = dict(color="#8c8c8c", linewidth=1.0)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#1f77b4", markeredgecolor="none", alpha=0.4),
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        if processed.div_start is not None:
            ax.set_xlabel("Organoid age (DIV)")
        else:
            ax.set_xlabel("Time point")
        ax.set_ylabel("Organoid ROI area (pixels)")
        ax.set_title("Organoid area distribution per timepoint")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)



def parse_args() -> argparse.Namespace:
    default_input = _default_roi_measurements_path()
    cfg = _load_config()

    parser = argparse.ArgumentParser(description="Plot organoid ROI size and growth curves with a consistent style.")
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

    area_plot_path = output_dir / f"{args.prefix}_area_over_time.png"
    growth_plot_path = output_dir / f"{args.prefix}_growth_curve.png"
    box_plot_path = output_dir / f"{args.prefix}_area_boxplot.png"

    _plot_area_over_time(processed, area_plot_path)
    _plot_growth_curve(processed, growth_plot_path)
    _plot_area_boxplots(processed, box_plot_path)

    print(f"Saved area plot to {area_plot_path}")
    print(f"Saved growth curve plot to {growth_plot_path}")
    print(f"Saved area box plot to {box_plot_path}")


if __name__ == "__main__":
    main()
