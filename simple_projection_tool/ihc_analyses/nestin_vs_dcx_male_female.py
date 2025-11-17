"""Male vs Female projection analysis for the Nestin/DCX dataset.

This subclass reuses the WT vs KO pipeline but groups samples by the inferred
sex token (``F``/``M``) embedded in the folder names. Outputs therefore land in
``analysis_results/NestinvsDcx_MaleVsFemale_IHC/`` and the resulting tables and
figures compare Female versus Male distributions instead of WT versus KO.
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
        """Group by the inferred sex token instead of genotype."""

        label = self._infer_subject_label(folder_name)
        mapping = {
            "F": "Female",
            "M": "Male",
        }
        return mapping.get(label)
