"""Male vs Female projection analysis for the Nestin/DCX dataset.

Unlike the genotype-based comparison, this helper simply asks whether the
sample folder contains an ``F`` or ``M`` token (case-insensitive) and then
funnels the projection statistics into the matching sex bucket. Everything else
is inherited from :class:`NestinvsDcx_WTvsKOIHCAnalysis`, so the analysis still
computes per-image pixel summaries, group statistics, and matched box/SEM plots,
but the labels read “Female” vs “Male”. Outputs land in
``analysis_results/NestinvsDcx_MaleVsFemale_IHC/analysis_pipeline/<channel>/`` in
the same structure as the WT vs KO run so downstream tooling can swap between
genotype and sex comparisons without extra coding.
"""

from __future__ import annotations

from typing import Optional

from .nestin_vs_dcx import NestinvsDcx_WTvsKOIHCAnalysis


class NestinvsDcx_MaleVsFemaleIHCAnalysis(NestinvsDcx_WTvsKOIHCAnalysis):
    """Compare Female and Male groups for the Nestin/DCX staining experiment."""

    name = "NestinvsDcx_MaleVsFemale_IHC"

    GROUP_LABEL = "Female vs Male"
    GROUP_NAMES = ("Female", "Male")
    GROUP_COLORS = {
        "Female": "#ff7f0e",
        "Male": "#1f77b4",
    }

    def _infer_group(self, folder_name: str) -> Optional[str]:
        """Group by the inferred sex token instead of genotype.

        The base implementation inspects folder names for IGI/IGIKO to detect WT
        vs KO runs. This override reuses the shared ``_infer_subject_label``
        helper and maps the resulting ``F``/``M`` code onto human-friendly
        strings. Any sample lacking a valid token is skipped so the comparison
        only contains folders with explicit sex metadata.
        """

        label = self._infer_subject_label(folder_name)
        mapping = {
            "F": "Female",
            "M": "Male",
        }
        return mapping.get(label)
