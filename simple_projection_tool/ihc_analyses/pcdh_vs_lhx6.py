"""WT vs KO projection analysis for the PCDH19/LHX6 immunostaining experiment.

This module contains the first concrete example of how to extend the
:mod:`ihc_analyses` framework. The analysis follows the structure laid out in
:class:`~ihc_analyses.base.ProjectionAnalysis` and documents every step so the
next biological question can reuse the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    path: Path


class PCDHvsLHX6_WTvsKOIHCAnalysis(ProjectionAnalysis):
    """Compare WT and KO groups for the PCDH19 vs LHX6 staining experiment.

    The analysis performs a complete superficial intensity review by:

    1. Importing all TIFFs produced by ``simple_channel_projections`` and
       determining whether each file belongs to the wild-type (``WT``) or
       knockout (``KO``) group based on the folder name (``IGI`` = WT,
       ``IGIKO`` = KO).
    2. Processing every 16-bit projection by loading the pixels into
       floating-point arrays (no scaling is applied) and computing per-image
       descriptive statistics (mean, median, maximum, standard deviation, pixel
       count, and a 95 percent confidence interval for the mean).
    3. Summarising statistics across groups, including Welch t-tests and
        Mann-Whitney U tests, so publication-ready values (``N``, means,
       medians, confidence intervals, p-values) are readily available.
    4. Plotting group-level distributions using matched conventions (boxplots and
       mean plus/minus SEM bar charts saved as SVG and PNG with Arial text) to
       give imaging scientists an immediate visual comparison for each
       projection type.

    All derived artefacts live beneath
    ``<base_path>/analysis_results/PCDHvsLHX6_WTvsKO_IHC/analysis_pipeline`` to
    reinforce their connection to this post-projection analysis layer.
    """

    #: Folder-friendly identifier for this analysis.
    name = "PCDHvsLHX6_WTvsKO_IHC"

    #: Expected suffixes (``_<suffix>.tif``) for the projection TIFFs. Keeping
    #: them in one place allows scientists to add new projection styles later.
    PROJECTION_SUFFIXES = {
        "max": "max",
        "mean": "mean",
        "median": "median",
    }

    #: 95 percent confidence interval using the normal approximation.
    CONFIDENCE_Z = 1.96

    def import_data(self) -> pd.DataFrame:
        """Catalogue the TIFF files that will be processed."""

        if not self.projection_root.exists():
            raise FileNotFoundError(
                f"Projection directory {self.projection_root} does not exist."
            )

        records: List[ProjectionRecord] = []
        for run_folder in sorted(
            path for path in self.projection_root.iterdir() if path.is_dir()
        ):
            group = self._infer_group(run_folder.name)
            if group is None:
                continue

            projections_dir = run_folder / "16bit"
            if not projections_dir.exists():
                continue

            for tif_path in sorted(projections_dir.glob("*.tif")):
                projection_type = self._infer_projection_type(tif_path.name)
                if projection_type is None:
                    continue
                records.append(
                    ProjectionRecord(
                        group=group,
                        sample_id=run_folder.name,
                        projection_type=projection_type,
                        path=tif_path,
                    )
                )

        if not records:
            raise FileNotFoundError(
                f"No projection TIFFs were discovered in {self.projection_root}."
            )

        manifest = pd.DataFrame(
            {
                "group": [record.group for record in records],
                "sample_id": [record.sample_id for record in records],
                "projection_type": [record.projection_type for record in records],
                "path": [record.path for record in records],
            }
        )
        return manifest

    def process_data(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Load each TIFF and compute per-image pixel statistics."""

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
                    "path": str(path),
                    **stats,
                }
            )

        results_df = pd.DataFrame(results)
        return results_df

    def run_statistics(self) -> None:
        """Calculate group-level summaries and hypothesis tests."""

        self.group_summary: Optional[pd.DataFrame] = None
        self.comparison_results: Optional[pd.DataFrame] = None

        if self.results is None or self.results.empty:
            return

        summary = self._summarise_groups(self.results)
        comparisons = self._compare_groups(self.results)

        summary_projection_types = (
            summary["projection_type"].nunique() if not summary.empty else 0
        )
        print(
            f"[{self.name}]     compiled group summaries for {summary_projection_types} projection type(s).",
            flush=True,
        )
        if comparisons.empty:
            print(
                f"[{self.name}]     no group comparisons were produced (missing WT/KO pairs).",
                flush=True,
            )
        else:
            comparison_projection_types = comparisons["projection_type"].nunique()
            print(
                f"[{self.name}]     calculated statistical tests for {comparison_projection_types} projection type(s).",
                flush=True,
            )

        self.group_summary = summary
        self.comparison_results = comparisons if not comparisons.empty else None

        self.register_table("group_summary", summary)
        if not comparisons.empty:
            self.register_table("group_comparisons", comparisons)

    def generate_plots(self) -> None:
        """Create summary figures for each projection type."""

        if self.results is None or self.results.empty:
            return

        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"

        for projection_type, projection_df in self.results.groupby("projection_type"):
            figure = self._plot_projection_summary(projection_type, projection_df)
            if figure is None:
                print(
                    f"[{self.name}]     skipped plotting for {projection_type!r} (missing WT/KO data)",
                    flush=True,
                )
                continue
            description = (
                "Pixel mean distribution (box plus mean and SEM) for WT versus KO projections"
            )
            metadata = {"Creator": self.name, "Description": description}
            self.save_figure(
                figure,
                f"{projection_type}_pixel_mean_summary",
                metadata=metadata,
            )
            print(
                f"[{self.name}]     finished plotting pixel mean summary for {projection_type!r} projections.",
                flush=True,
            )
            plt.close(figure)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_group(self, folder_name: str) -> Optional[str]:
        """Classify a folder as WT or KO based on its name."""

        lowered = folder_name.lower()
        if "igiko" in lowered:
            return "KO"
        if lowered == "igi" or lowered.startswith("igi_") or lowered.startswith("igi-"):
            return "WT"
        if "igi" in lowered and "igiko" not in lowered:
            return "WT"
        return None

    def _infer_projection_type(self, filename: str) -> Optional[str]:
        """Work out whether a TIFF is max/mean/median based on its suffix."""

        lowered = filename.lower()
        for label in self.PROJECTION_SUFFIXES:
            if lowered.endswith(f"_{label}.tif"):
                return label
        return None

    def _load_projection(self, path: Path) -> np.ndarray:
        """Read a TIFF into memory as a 2-D floating point array."""

        array = tiff.imread(path)
        if array.ndim != 2:
            array = np.squeeze(array)
        return array.astype(np.float64, copy=False)

    def _compute_statistics(self, array: np.ndarray) -> Dict[str, float]:
        """Calculate descriptive statistics for the supplied projection."""

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
        """Aggregate per-image measurements into group-level summaries."""

        summaries: List[Dict[str, float | str | int]] = []
        for projection_type, projection_df in results.groupby("projection_type"):
            for group, group_df in projection_df.groupby("group"):
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
                        "n": n,
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
        """Run parametric and non-parametric tests for each projection type."""

        try:
            from scipy import stats  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "scipy is required to run statistical tests. Install it in the analysis environment."
            ) from exc

        comparisons: List[Dict[str, float | str | int]] = []
        for projection_type, projection_df in results.groupby("projection_type"):
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
                }
            )

        return pd.DataFrame(comparisons)

    def _plot_projection_summary(
        self,
        projection_type: str,
        projection_df: pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """Return a figure showing WT/KO intensity comparisons for one projection."""

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

        # Box plot of per-image means
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

        # Bar chart with mean +/- SEM
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
            f"WT vs KO pixel mean comparison - {projection_type.upper()} projection",
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
