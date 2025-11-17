"""WT vs KO analysis stratified by sex for the Nestin/DCX dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .nestin_vs_dcx import NestinvsDcx_WTvsKOIHCAnalysis


class NestinvsDcx_WTvsKOBySexIHCAnalysis(NestinvsDcx_WTvsKOIHCAnalysis):
    """Compare WT vs KO separately for Female and Male samples."""

    name = "NestinvsDcx_WTvsKO_BySex_IHC"

    SEX_ORDER = ("F", "M")
    SEX_NAMES = {
        "F": "Female",
        "M": "Male",
    }

    def __init__(
        self,
        base_path: Path | str,
        *,
        projection_dir_name: str = "simple_projections",
        output_dir: Optional[Path | str] = None,
        channel_filter: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            base_path,
            projection_dir_name=projection_dir_name,
            output_dir=output_dir,
            channel_filter=channel_filter,
        )
        self._current_subject_label_filter: Optional[str] = None

    def run(self) -> None:
        aggregated_tables: List[Path] = []
        aggregated_figures: List[Path] = []
        aggregated_per_images: List[Path] = []

        original_filter = self._current_subject_label_filter
        original_pipeline_dir = self.pipeline_dir

        for label in self.SEX_ORDER:
            self._current_subject_label_filter = label
            sex_name = self.SEX_NAMES.get(label, label)
            self._configure_pipeline_dirs()
            print(
                f"[{self.name}] ============================== sex {sex_name} ==============================",
                flush=True,
            )
            try:
                super().run()
            except FileNotFoundError as exc:
                print(f"[{self.name}]     Skipping sex {sex_name}: {exc}", flush=True)
                continue

            aggregated_tables.extend(self.saved_table_paths)
            aggregated_figures.extend(self.saved_figure_paths)
            aggregated_per_images.extend(self.per_image_summary_paths)

        self._current_subject_label_filter = original_filter
        self._configure_pipeline_dirs()
        self.pipeline_dir = getattr(self, "pipeline_root", original_pipeline_dir)
        self.saved_table_paths = aggregated_tables
        self.saved_figure_paths = aggregated_figures
        self.per_image_summary_paths = aggregated_per_images

    def import_data(self) -> pd.DataFrame:
        manifest = super().import_data()
        label = self._current_subject_label_filter
        if label is None:
            return manifest
        filtered = manifest.loc[manifest["subject_label"] == label]
        if filtered.empty:
            raise FileNotFoundError(
                f"No projection TIFFs found for subject label {label!r} "
                f"in {self.projection_root}."
            )
        return filtered

    def _derive_channel_context_slug(self) -> str:
        slug = super()._derive_channel_context_slug()
        if self._current_subject_label_filter:
            sex_name = self.SEX_NAMES.get(self._current_subject_label_filter, "unknown")
            slug = f"{self._slugify(sex_name)}_{slug}"
        return slug
