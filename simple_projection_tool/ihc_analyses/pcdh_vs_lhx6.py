"""WT vs KO projection analysis for the PCDH19/LHX6 immunostaining experiment.

This module contains the first concrete example of how to extend the
:mod:`ihc_analyses` framework.  The analysis follows the structure laid out in
:class:`~ihc_analyses.base.ProjectionAnalysis` and documents every step so the
next biological question can reuse the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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

    The analysis performs three main tasks:

    1. **Import** – find all TIFFs produced by ``simple_channel_projections``
       under ``<base_path>/simple_projections`` and determine whether each file
       belongs to the wild-type (``WT``) or knockout (``KO``) group based on the
        folder name (``IGI`` = WT, ``IGIKO`` = KO).
    2. **Process** – load each 16-bit TIFF, compute descriptive statistics for
       the pixel intensities (mean, median, max, standard deviation, and a 95 %
       confidence interval), and collect the results in a DataFrame.
    3. **Save** – rely on the base-class helper to export the manifest and
       results to ``analysis_results/PCDHvsLHX6_WTvsKO_IHC`` unless another
       output directory is specified.

    Subsequent analyses can copy this file, change the group-identification
    rules, and insert additional statistics or plots as required.
    """

    #: Folder-friendly identifier for this analysis.
    name = "PCDHvsLHX6_WTvsKO_IHC"

    #: Expected suffixes (``_<suffix>.tif``) for the projection TIFFs.  Keeping
    #: them in one place allows scientists to add new projection styles later.
    PROJECTION_SUFFIXES = {
        "max": "max",
        "mean": "mean",
        "median": "median",
    }

    #: 95 % confidence interval using the normal approximation.  Replace with a
    #: t-distribution or bootstrap method if future experiments require it.
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
            # Determine whether the folder represents WT or KO.  If we cannot
            # tell (e.g. the folder is a calibration run), skip it gracefully.
            group = self._infer_group(run_folder.name)
            if group is None:
                continue

            projections_dir = run_folder / "16bit"
            if not projections_dir.exists():
                # Some runs may be incomplete; note the omission by skipping.
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
            path: Path = Path(row.path)
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
            # Catch variations such as ``igi sample1``.
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
            # Remove singleton dimensions left by some export tools.
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

    def _confidence_interval(
        self, mean: float, std: float, count: float
    ) -> tuple[float, float]:
        """Return the 95 % confidence interval for the mean pixel intensity."""

        if count <= 1 or std == 0.0:
            # With a single pixel or no variation, the CI collapses to the mean.
            return (mean, mean)
        standard_error = std / np.sqrt(count)
        margin = self.CONFIDENCE_Z * standard_error
        return (mean - margin, mean + margin)

