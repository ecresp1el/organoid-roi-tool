"""WT vs KO projection analysis for the Nestin/DCX immunostaining experiment.

Design principles mirrored from ``pcdh_vs_lhx6`` so new analyses can be created
quickly:

* The base class verifies the presence of ``simple_projections`` and regenerates
  them when needed, so the CLI remains a single turnkey command per dataset.
* ``CHANNEL_ALIASES`` and ``CHANNEL_METADATA`` describe the markers available in
  this analysis (Nestin, DCX, and optionally PCDH19). Adjust these dictionaries
  when porting the template to new markers.
* Each requested channel runs as an independent pass. Results land under
  ``analysis_results/NestinvsDcx_WTvsKO_IHC/analysis_pipeline/<channel>/``
  keeping figures and CSVs separated by marker.
* The manifest/results tables include ``subject_label`` and channel metadata
  columns so downstream comparisons retain both sex and marker context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff  # type: ignore

from .base import ProjectionAnalysis


@dataclass(frozen=True)
class ProjectionRecord:
    """Describe a single TIFF projection and its experimental context."""

    group: str
    sample_id: str
    projection_type: str
    channel: str
    channel_canonical: str
    channel_marker: str
    channel_wavelength_nm: Optional[float]
    subject_label: Optional[str]
    path: Path


class NestinvsDcx_WTvsKOIHCAnalysis(ProjectionAnalysis):
    """Compare WT and KO groups for the Nestin/DCX staining experiment."""

    name = "NestinvsDcx_WTvsKO_IHC"

    PROJECTION_SUFFIXES = {
        "max": "max",
        "mean": "mean",
        "median": "median",
    }

    CONFIDENCE_Z = 1.96

    CHANNEL_ALIASES = {
        "nestin": (
            "Nestin",
            "Confocal - Green",
            "Confocal_-_Green",
            "confocal_-_green",
            "confocal_green",
            "green",
        ),
        "dcx": (
            "DCX",
            "Confocal - Far red",
            "Confocal_-_Far_red",
            "confocal_-_far_red",
            "confocal_far_red",
            "far_red",
        ),
        "pcdh19": (
            "PCDH19",
            "Confocal - Red",
            "Confocal_-_Red",
            "confocal_-_red",
            "confocal_red",
            "red",
        ),
    }

    CHANNEL_METADATA = {
        "nestin": {
            "marker": "Nestin",
            "canonical_name": "Confocal - Green",
            "wavelength_nm": 529.0,
        },
        "dcx": {
            "marker": "DCX",
            "canonical_name": "Confocal - Far red",
            "wavelength_nm": 700.0,
        },
        "pcdh19": {
            "marker": "PCDH19",
            "canonical_name": "Confocal - Red",
            "wavelength_nm": 600.0,
        },
    }

    def __init__(
        self,
        base_path: Path | str,
        *,
        projection_dir_name: str = "simple_projections",
        output_dir: Optional[Path | str] = None,
        channel_filter: Optional[Sequence[str]] = None,
    ) -> None:
        requested = (
            tuple(channel_filter)
            if channel_filter is not None
            else ("Nestin", "DCX", "PCDH19")
        )
        expanded = self._expand_channel_aliases(requested)
        super().__init__(
            base_path,
            projection_dir_name=projection_dir_name,
            output_dir=output_dir,
            channel_filter=expanded,
        )
        self.channel_filter_names = requested
        self._configure_pipeline_dirs()

    def run(self) -> None:
        unique_channels: list[str] = []
        seen: set[str] = set()
        for name in self.channel_filter_names:
            key = self._normalise_channel_name(name)
            if key not in seen:
                unique_channels.append(name)
                seen.add(key)

        if len(unique_channels) <= 1:
            super().run()
            return

        original_names = self.channel_filter_names
        original_filter = self.channel_filter.copy() if self.channel_filter else None
        aggregated_tables: list[Path] = []
        aggregated_figures: list[Path] = []

        for name in unique_channels:
            aliases = self._expand_channel_aliases((name,))
            channel_filter_set = {
                self._normalise_channel_name(alias) for alias in aliases if alias
            }
            self.channel_filter_names = (name,)
            self.channel_filter = channel_filter_set or None
            self._configure_pipeline_dirs()
            print(
                f"[{self.name}] ============================== channel {name} ==============================",
                flush=True,
            )
            try:
                super().run()
            except FileNotFoundError as exc:
                print(f"[{self.name}]     Skipping channel {name}: {exc}", flush=True)
                continue
            aggregated_tables.extend(self.saved_table_paths)
            aggregated_figures.extend(self.saved_figure_paths)

        self.channel_filter_names = original_names
        self.channel_filter = original_filter
        self._configure_pipeline_dirs()
        self.pipeline_dir = getattr(self, "pipeline_root", self.pipeline_dir)
        self.saved_table_paths = aggregated_tables
        self.saved_figure_paths = aggregated_figures

    def _expand_channel_aliases(self, names: Sequence[str]) -> tuple[str, ...]:
        expanded: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not name:
                continue
            if name not in seen:
                expanded.append(name)
                seen.add(name)
            alias_key = self._normalise_channel_name(name)
            for alias in self.CHANNEL_ALIASES.get(alias_key, ()):
                if alias not in seen:
                    expanded.append(alias)
                    seen.add(alias)
        return tuple(expanded)

    def import_data(self) -> pd.DataFrame:
        if not self.projection_root.exists():
            raise FileNotFoundError(
                f"Projection directory {self.projection_root} does not exist."
            )

        records: List[ProjectionRecord] = []
        discovered_channels: set[str] = set()
        for run_folder in sorted(
            path for path in self.projection_root.iterdir() if path.is_dir()
        ):
            group = self._infer_group(run_folder.name)
            if group is None:
                continue
            subject_label = self._infer_subject_label(run_folder.name)

            projections_dir = run_folder / "16bit"
            if not projections_dir.exists():
                continue

            for tif_path in sorted(projections_dir.glob("*.tif")):
                projection_type = self._infer_projection_type(tif_path.name)
                if projection_type is None:
                    continue
                channel = self._extract_channel_name(tif_path.name)
                discovered_channels.add(channel)
                if not self._channel_is_selected(channel):
                    continue
                canonical_key = self._canonical_channel_key(channel)
                metadata = self.CHANNEL_METADATA.get(canonical_key or "", {})
                records.append(
                    ProjectionRecord(
                        group=group,
                        sample_id=run_folder.name,
                        projection_type=projection_type,
                        channel=channel,
                        channel_canonical=metadata.get("canonical_name", channel),
                        channel_marker=metadata.get("marker", canonical_key or channel),
                        channel_wavelength_nm=metadata.get("wavelength_nm"),
                        subject_label=subject_label,
                        path=tif_path,
                    )
                )

        if not records:
            selected = (
                ", ".join(self.channel_filter_names)
                if self.channel_filter_names
                else "the specified channels"
            )
            available = ", ".join(sorted(discovered_channels)) or "none"
            raise FileNotFoundError(
                f"No projection TIFFs were discovered in {self.projection_root} "
                f"for channel selection: {selected}. Available channels: {available}."
            )

        manifest = pd.DataFrame(
            {
                "group": [record.group for record in records],
                "sample_id": [record.sample_id for record in records],
                "projection_type": [record.projection_type for record in records],
                "channel": [record.channel for record in records],
                "channel_canonical": [record.channel_canonical for record in records],
                "channel_marker": [record.channel_marker for record in records],
                "channel_wavelength_nm": [record.channel_wavelength_nm for record in records],
                "subject_label": [record.subject_label for record in records],
                "path": [record.path for record in records],
            }
        )
        return manifest

    def process_data(self, manifest: pd.DataFrame) -> pd.DataFrame:
        results: List[dict] = []
        for row in manifest.itertuples():
            path = Path(row.path)
            data = self._load_projection(path)
            stats = self._compute_statistics(data)
            results.append(
                {
                    "group": row.group,
                    "sample_id": row.sample_id,
                    "projection_type": row.projection_type,
                    "channel": row.channel,
                    "channel_canonical": row.channel_canonical,
                    "channel_marker": row.channel_marker,
                    "channel_wavelength_nm": row.channel_wavelength_nm,
                    "subject_label": row.subject_label,
                    "path": str(path),
                    **stats,
                }
            )

        return pd.DataFrame(results)

    def run_statistics(self) -> None:
        self.group_summary = None
        self.comparison_results = None

        if self.results is None or self.results.empty:
            return

        summary = self._summarise_groups(self.results)
        comparisons = self._compare_groups(self.results)

        if summary.empty:
            print(
                f"[{self.name}]     group summaries are empty (no eligible projections).",
                flush=True,
            )
        else:
            summary_channels = ", ".join(sorted(summary["channel_canonical"].unique()))
            print(
                f"[{self.name}]     compiled WT vs KO summaries per projection type for channel(s): {summary_channels}.",
                flush=True,
            )
        if comparisons.empty:
            print(
                f"[{self.name}]     no group comparisons were produced (missing WT/KO pairs).",
                flush=True,
            )
        else:
            comparison_channels = ", ".join(sorted(comparisons["channel_canonical"].unique()))
            print(
                f"[{self.name}]     calculated WT vs KO statistical tests per projection type for channel(s): {comparison_channels}.",
                flush=True,
            )

        self.group_summary = summary
        self.comparison_results = comparisons if not comparisons.empty else None

        self.register_table("group_summary", summary)
        if not comparisons.empty:
            self.register_table("group_comparisons", comparisons)

    def generate_plots(self) -> None:
        if self.results is None or self.results.empty:
            return

        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"

        for (channel, projection_type), projection_df in self.results.groupby(
            ["channel", "projection_type"]
        ):
            figure = self._plot_projection_summary(channel, projection_type, projection_df)
            if figure is None:
                print(
                    f"[{self.name}]     skipped plotting for {channel!r} / {projection_type!r} (missing WT/KO data)",
                    flush=True,
                )
                continue

            description = (
                "Pixel mean distribution (box plus mean and SEM) for WT versus KO projections "
                f"[{projection_df['channel_marker'].iloc[0]} – {projection_df['channel_canonical'].iloc[0]}]"
            )
            metadata = {
                "Creator": self.name,
                "Description": description,
                "Channel": projection_df["channel_canonical"].iloc[0],
                "Marker": projection_df["channel_marker"].iloc[0],
            }
            wavelength = projection_df["channel_wavelength_nm"].dropna().unique()
            if wavelength.size == 1:
                metadata["Wavelength_nm"] = f"{float(wavelength[0]):.1f}"

            self.save_figure(
                figure,
                f"{channel}_{projection_type}_pixel_mean_summary",
                metadata=metadata,
            )
            print(
                f"[{self.name}]     finished plotting pixel mean summary for {channel!r} / {projection_type!r} projections.",
                flush=True,
            )
            plt.close(figure)

    def _infer_group(self, folder_name: str) -> Optional[str]:
        lowered = folder_name.lower()
        if "igiko" in lowered:
            return "KO"
        if lowered == "igi" or lowered.startswith("igi_") or lowered.startswith("igi-"):
            return "WT"
        if "igi" in lowered and "igiko" not in lowered:
            return "WT"
        return None

    def _infer_projection_type(self, filename: str) -> Optional[str]:
        lowered = filename.lower()
        for label in self.PROJECTION_SUFFIXES:
            if lowered.endswith(f"_{label}.tif"):
                return label
        return None

    def _load_projection(self, path: Path) -> np.ndarray:
        array = tiff.imread(path)
        if array.ndim != 2:
            array = np.squeeze(array)
        return array.astype(np.float64, copy=False)

    def _compute_statistics(self, array: np.ndarray) -> Dict[str, float]:
        flattened = array.ravel()
        pixel_count = float(flattened.size)
        pixel_mean = float(np.mean(flattened))
        pixel_median = float(np.median(flattened))
        pixel_std = float(np.std(flattened, ddof=1)) if flattened.size > 1 else 0.0
        pixel_max = float(np.max(flattened))
        ci_low, ci_high = self._confidence_interval(pixel_mean, pixel_std, pixel_count)
        return {
            "pixel_count": pixel_count,
            "pixel_mean": pixel_mean,
            "pixel_median": pixel_median,
            "pixel_std": pixel_std,
            "pixel_max": pixel_max,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    def _summarise_groups(self, results: pd.DataFrame) -> pd.DataFrame:
        summaries: List[Dict[str, float | str | int]] = []
        for (channel, projection_type), projection_df in results.groupby(
            ["channel", "projection_type"]
        ):
            meta_row = projection_df.iloc[0]
            for group, group_df in projection_df.groupby("group"):
                subject_labels = sorted(
                    {label for label in group_df["subject_label"].dropna().astype(str)}
                )
                pixel_means = group_df["pixel_mean"].astype(float).to_numpy()
                pixel_medians = group_df["pixel_median"].astype(float).to_numpy()
                n = int(pixel_means.size)
                if n == 0:
                    continue
                mean_mean = float(np.mean(pixel_means))
                median_mean = float(np.median(pixel_means))
                std_mean = float(np.std(pixel_means, ddof=1)) if n > 1 else 0.0
                sem = std_mean / np.sqrt(n) if n > 0 else float("nan")
                ci_low, ci_high = self._confidence_interval(mean_mean, std_mean, float(n))
                summaries.append(
                    {
                        "analysis": self.name,
                        "projection_type": projection_type,
                        "group": group,
                        "channel": channel,
                        "channel_canonical": meta_row.get("channel_canonical", channel),
                        "channel_marker": meta_row.get("channel_marker", channel),
                        "channel_wavelength_nm": meta_row.get("channel_wavelength_nm"),
                        "n": n,
                        "subject_labels": ";".join(subject_labels) if subject_labels else "",
                        "pixel_mean_mean": mean_mean,
                        "pixel_mean_median": median_mean,
                        "pixel_mean_std": std_mean,
                        "pixel_mean_sem": sem,
                        "pixel_mean_ci_low": ci_low,
                        "pixel_mean_ci_high": ci_high,
                        "pixel_median_mean": float(np.mean(pixel_medians)),
                        "pixel_median_median": float(np.median(pixel_medians)),
                    }
                )

        return pd.DataFrame(summaries)

    def _compare_groups(self, results: pd.DataFrame) -> pd.DataFrame:
        try:
            from scipy import stats  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required to run statistical tests. Install it in the analysis environment."
            ) from exc

        comparisons: List[Dict[str, float | str | int]] = []
        for (channel, projection_type), projection_df in results.groupby(
            ["channel", "projection_type"]
        ):
            meta_row = projection_df.iloc[0]
            wt_values = (
                projection_df.loc[projection_df["group"] == "WT", "pixel_mean"]
                .astype(float)
                .to_numpy()
            )
            ko_values = (
                projection_df.loc[projection_df["group"] == "KO", "pixel_mean"]
                .astype(float)
                .to_numpy()
            )

            if wt_values.size == 0 or ko_values.size == 0:
                continue

            ttest = stats.ttest_ind(wt_values, ko_values, equal_var=False, nan_policy="omit")
            mannwhitney = stats.mannwhitneyu(
                wt_values,
                ko_values,
                alternative="two-sided",
            )

            def _sem(values: np.ndarray) -> float:
                if values.size <= 1:
                    return 0.0
                return float(np.std(values, ddof=1) / np.sqrt(values.size))

            comparisons.append(
                {
                    "analysis": self.name,
                    "projection_type": projection_type,
                    "channel": channel,
                    "channel_canonical": meta_row.get("channel_canonical", channel),
                    "channel_marker": meta_row.get("channel_marker", channel),
                    "channel_wavelength_nm": meta_row.get("channel_wavelength_nm"),
                    "metric": "pixel_mean",
                    "parametric_test": "Welch t-test",
                    "parametric_statistic": float(ttest.statistic),
                    "parametric_pvalue": float(ttest.pvalue),
                    "nonparametric_test": "Mann-Whitney U",
                    "nonparametric_statistic": float(mannwhitney.statistic),
                    "nonparametric_pvalue": float(mannwhitney.pvalue),
                    "wt_n": int(wt_values.size),
                    "ko_n": int(ko_values.size),
                    "wt_mean": float(np.mean(wt_values)),
                    "ko_mean": float(np.mean(ko_values)),
                    "wt_median": float(np.median(wt_values)),
                    "ko_median": float(np.median(ko_values)),
                    "wt_sem": _sem(wt_values),
                    "ko_sem": _sem(ko_values),
                    "subject_labels": ";".join(
                        sorted(
                            {label for label in projection_df["subject_label"].dropna().astype(str)}
                        )
                    ),
                }
            )

        return pd.DataFrame(comparisons)

    def _plot_projection_summary(
        self,
        channel: str,
        projection_type: str,
        projection_df: pd.DataFrame,
    ) -> Optional[plt.Figure]:
        groups = ["WT", "KO"]
        colors = {"WT": "#1f77b4", "KO": "#d62728"}
        values = [
            projection_df.loc[projection_df["group"] == group, "pixel_mean"]
            .astype(float)
            .to_numpy()
            for group in groups
        ]

        if any(arr.size == 0 for arr in values):
            return None

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        box = axes[0].boxplot(
            values,
            labels=groups,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for patch, group in zip(box["boxes"], groups):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.6)
        axes[0].set_title(f"{projection_type.upper()} projection: pixel mean")
        axes[0].set_ylabel("Pixel mean intensity (a.u.)")
        axes[0].grid(alpha=0.3)

        scatter_x: list[float] = []
        scatter_y: list[float] = []
        for idx, arr in enumerate(values, start=1):
            if arr.size == 0:
                continue
            arr = arr.astype(float)
            jitter = np.linspace(-0.12, 0.12, arr.size) if arr.size > 1 else np.array([0.0])
            scatter_x.extend((idx + jitter).tolist())
            scatter_y.extend(arr.tolist())
        if scatter_x:
            axes[0].scatter(
                scatter_x,
                scatter_y,
                color="black",
                alpha=0.35,
                s=22,
                linewidths=0,
                zorder=3,
            )

        means = [float(np.mean(arr)) for arr in values]
        sems = [
            float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            for arr in values
        ]
        x = np.arange(len(groups))
        axes[1].bar(
            x,
            means,
            yerr=sems,
            color=[colors[group] for group in groups],
            alpha=0.8,
            capsize=8,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(groups)
        axes[1].set_title("Mean +/- SEM (pixel mean)")
        axes[1].grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{projection_df['channel_marker'].iloc[0]} – WT vs KO pixel mean comparison "
            f"({projection_type.upper()} projection)",
            fontsize=14,
        )
        fig.tight_layout()

        return fig

    def _confidence_interval(
        self,
        mean: float,
        std: float,
        count: float,
    ) -> tuple[float, float]:
        """Return the 95 percent confidence interval for the mean pixel intensity."""

        if count <= 1 or std == 0.0:
            return (mean, mean)
        standard_error = std / np.sqrt(count)
        margin = self.CONFIDENCE_Z * standard_error
        return (mean - margin, mean + margin)

    @staticmethod
    def _extract_channel_name(filename: str) -> str:
        stem = Path(filename).stem
        if "_" not in stem:
            return stem
        channel, _suffix = stem.rsplit("_", 1)
        return channel or stem

    @staticmethod
    def _infer_subject_label(folder_name: str) -> Optional[str]:
        import re

        tokens = [token for token in re.split(r"[^A-Za-z0-9]+", folder_name.upper()) if token]
        if any(token in {"F", "FEMALE", "FEM"} for token in tokens):
            return "F"
        if any(token in {"M", "MALE"} for token in tokens):
            return "M"
        return None

    def _canonical_channel_key(self, channel: str) -> Optional[str]:
        normalised = self._normalise_channel_name(channel)
        for canonical, aliases in self.CHANNEL_ALIASES.items():
            alias_set = {self._normalise_channel_name(alias) for alias in aliases}
            alias_set.add(canonical)
            if normalised in alias_set:
                return canonical
        if normalised in self.CHANNEL_METADATA:
            return normalised
        return None
